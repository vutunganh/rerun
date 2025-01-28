#![allow(dead_code, unused_variables, clippy::unnecessary_wraps)]

use std::sync::{atomic::AtomicBool, Arc};

use re_video::{decode::FrameContent, Chunk, Frame, Time};

use parking_lot::Mutex;

use crate::{
    resource_managers::SourceImageDataFormat,
    video::{
        player::{TimedDecodingError, VideoTexture},
        VideoPlayerError,
    },
    wgpu_resources::GpuTexture,
    RenderContext,
};

#[derive(Default)]
struct DecoderOutput {
    frames: Vec<Frame>,

    /// Set on error; reset on success.
    error: Option<TimedDecodingError>,
}

/// Internal implementation detail of the [`super::player::VideoPlayer`].
// TODO(andreas): Meld this into `super::player::VideoPlayer`.
pub struct VideoChunkDecoder {
    decoder: Box<dyn re_video::decode::AsyncDecoder>,
    decoder_output: Arc<Mutex<DecoderOutput>>,

    /// Set to true if the video texture has caught up with the desired presentation timestamp since the last reset.
    video_texture_up_to_date_since_last_reset: AtomicBool,
}

impl VideoChunkDecoder {
    pub fn new(
        debug_name: String,
        make_decoder: impl FnOnce(
            Box<dyn Fn(re_video::decode::Result<Frame>) + Send + Sync>,
        )
            -> re_video::decode::Result<Box<dyn re_video::decode::AsyncDecoder>>,
    ) -> Result<Self, VideoPlayerError> {
        re_tracing::profile_function!();

        let decoder_output = Arc::new(Mutex::new(DecoderOutput::default()));

        let on_output = {
            let decoder_output = decoder_output.clone();
            move |frame: re_video::decode::Result<Frame>| match frame {
                Ok(frame) => {
                    re_log::trace!(
                        "Decoded frame at PTS {:?}",
                        frame.info.presentation_timestamp
                    );
                    let mut output = decoder_output.lock();
                    output.frames.push(frame);
                    output.error = None; // We successfully decoded a frame, reset the error state.
                }
                Err(err) => {
                    // Many of the errors we get from a decoder are recoverable.
                    // They may be very frequent, but it's still useful to see them in the debug log for troubleshooting.
                    re_log::debug_once!("Error during decoding of {debug_name}: {err}");

                    let err = VideoPlayerError::Decoding(err);
                    let mut output = decoder_output.lock();
                    if let Some(error) = &mut output.error {
                        error.latest_error = err;
                    } else {
                        output.error = Some(TimedDecodingError::new(err));
                    }
                }
            }
        };

        let decoder = make_decoder(Box::new(on_output))?;

        Ok(Self {
            decoder,
            decoder_output,
            video_texture_up_to_date_since_last_reset: AtomicBool::new(false),
        })
    }

    /// Start decoding the given chunk.
    pub fn decode(&mut self, chunk: Chunk) -> Result<(), VideoPlayerError> {
        self.decoder.submit_chunk(chunk)?;
        Ok(())
    }

    /// Called after submitting the last chunk.
    ///
    /// Should flush all pending frames.
    pub fn end_of_video(&mut self) -> Result<(), VideoPlayerError> {
        self.decoder.end_of_video()?;
        Ok(())
    }

    /// Get the latest decoded frame at the given time
    /// and copy it to the given texture.
    ///
    /// Drop all earlier frames to save memory.
    ///
    /// Returns [`VideoPlayerError::EmptyBuffer`] if the internal buffer is empty,
    /// which it is just after startup or after a call to [`Self::reset`].
    pub fn update_video_texture(
        &self,
        render_ctx: &RenderContext,
        video_texture: &mut VideoTexture,
        presentation_timestamp: Time,
    ) -> Result<(), VideoPlayerError> {
        let mut decoder_output = self.decoder_output.lock();
        let frames = &mut decoder_output.frames;

        let Some(frame_idx) = re_video::demux::latest_at_idx(
            frames,
            |frame| frame.info.presentation_timestamp,
            &presentation_timestamp,
        ) else {
            return Err(VideoPlayerError::EmptyBuffer);
        };

        // drain up-to (but not including) the frame idx, clearing out any frames
        // before it. this lets the video decoder output more frames.
        drop(frames.drain(0..frame_idx));

        // after draining all old frames, the next frame will be at index 0
        let frame_idx = 0;
        let frame = &frames[frame_idx];

        let texture_frame_info = video_texture.frame_info.as_ref();
        let new_frame_time_range = frame.info.presentation_time_range();

        let texture_is_already_up_to_date = texture_frame_info
            .is_some_and(|info| info.presentation_time_range() == new_frame_time_range);
        let new_frame_is_exact_frame_requested =
            new_frame_time_range.contains(&presentation_timestamp);

        // If we're outdated & the frame is the one we asked for, take it!
        //
        // But if we're outdated but the incoming frame isn't the one we asked for it's a bit more nuanced:
        // If the decoder is just lagging behind a little bit during playback, we definitely want to show that new frame.
        // But if there was a seek by the user, we can get very awkward behavior if we show the frames as the come in by the decoder.
        // (since decoders can only jump to certain frames and have to catch up from there, there will be a lot of frames that are too old!)
        // Therefore, we only show frames if we previously caught up.
        // Since any jump backwards and any jump forward beyond what's enqueued will cause a reset, we should never run into the described problematic situation.
        let video_texture_up_to_date_since_last_reset = self
            .video_texture_up_to_date_since_last_reset
            .load(std::sync::atomic::Ordering::Acquire);
        let is_better_than_current = !texture_is_already_up_to_date
            && (new_frame_is_exact_frame_requested || video_texture_up_to_date_since_last_reset);

        if is_better_than_current {
            #[cfg(target_arch = "wasm32")]
            {
                video_texture.source_pixel_format = copy_web_video_frame_to_texture(
                    render_ctx,
                    &frame.content,
                    &video_texture.texture,
                )?;
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                video_texture.source_pixel_format = copy_native_video_frame_to_texture(
                    render_ctx,
                    &frame.content,
                    &video_texture.texture,
                )?;
            }

            video_texture.frame_info = Some(frame.info.clone());

            if new_frame_is_exact_frame_requested {
                self.video_texture_up_to_date_since_last_reset
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }

        Ok(())
    }

    /// Reset the video decoder and discard all frames.
    pub fn reset(&mut self) -> Result<(), VideoPlayerError> {
        self.decoder.reset()?;

        let mut decoder_output = self.decoder_output.lock();
        decoder_output.error = None;
        decoder_output.frames.clear();
        *self.video_texture_up_to_date_since_last_reset.get_mut() = false;

        Ok(())
    }

    /// Return and clear the latest error that happened during decoding.
    pub fn take_error(&self) -> Option<TimedDecodingError> {
        self.decoder_output.lock().error.take()
    }
}

#[cfg(target_arch = "wasm32")]
fn copy_web_video_frame_to_texture(
    ctx: &RenderContext,
    frame: &FrameContent,
    target_texture: &GpuTexture,
) -> Result<SourceImageDataFormat, VideoPlayerError> {
    let size = wgpu::Extent3d {
        width: frame.display_width(),
        height: frame.display_height(),
        depth_or_array_layers: 1,
    };
    let frame: &web_sys::VideoFrame = frame;
    let source = wgpu::CopyExternalImageSourceInfo {
        // Careful: `web_sys::VideoFrame` has a custom `clone` method:
        // https://developer.mozilla.org/en-US/docs/Web/API/VideoFrame/clone
        // We instead just want to clone the js value wrapped in VideoFrame!
        source: wgpu::ExternalImageSource::VideoFrame(Clone::clone(frame)),
        origin: wgpu::Origin2d { x: 0, y: 0 },
        flip_y: false,
    };
    let dest = wgpu::CopyExternalImageDestInfo {
        texture: &target_texture.texture,
        mip_level: 0,
        origin: wgpu::Origin3d { x: 0, y: 0, z: 0 },
        aspect: wgpu::TextureAspect::All,
        color_space: wgpu::PredefinedColorSpace::Srgb,
        premultiplied_alpha: false,
    };

    ctx.queue
        .copy_external_image_to_texture(&source, dest, size);

    Ok(SourceImageDataFormat::WgpuCompatible(
        target_texture.creation_desc.format,
    ))
}

#[cfg(not(target_arch = "wasm32"))]
fn copy_native_video_frame_to_texture(
    ctx: &RenderContext,
    frame: &FrameContent,
    target_texture: &GpuTexture,
) -> Result<SourceImageDataFormat, VideoPlayerError> {
    use crate::resource_managers::{
        transfer_image_data_to_texture, ImageDataDesc, SourceImageDataFormat,
        YuvMatrixCoefficients, YuvPixelLayout, YuvRange,
    };

    let format = match frame.format {
        re_video::PixelFormat::Rgb8Unorm => {
            // TODO(andreas): `ImageDataDesc` should have RGB handling!
            return copy_native_video_frame_to_texture(
                ctx,
                &FrameContent {
                    data: crate::pad_rgb_to_rgba(&frame.data, 255_u8),
                    format: re_video::PixelFormat::Rgba8Unorm,
                    ..*frame
                },
                target_texture,
            );
        }
        re_video::PixelFormat::Rgba8Unorm | re_video::PixelFormat::Yuv { .. } => {
            wgpu::TextureFormat::Rgba8Unorm
        }
    };

    re_tracing::profile_function!();

    let format = match &frame.format {
        re_video::PixelFormat::Rgb8Unorm => {
            unreachable!("Handled explicitly earlier in this function");
        }

        re_video::PixelFormat::Rgba8Unorm => {
            SourceImageDataFormat::WgpuCompatible(wgpu::TextureFormat::Rgba8Unorm)
        }

        re_video::PixelFormat::Yuv {
            layout,
            range,
            coefficients,
        } => SourceImageDataFormat::Yuv {
            layout: match layout {
                re_video::decode::YuvPixelLayout::Y_U_V444 => YuvPixelLayout::Y_U_V444,
                re_video::decode::YuvPixelLayout::Y_U_V422 => YuvPixelLayout::Y_U_V422,
                re_video::decode::YuvPixelLayout::Y_U_V420 => YuvPixelLayout::Y_U_V420,
                re_video::decode::YuvPixelLayout::Y400 => YuvPixelLayout::Y400,
            },
            coefficients: match coefficients {
                re_video::decode::YuvMatrixCoefficients::Identity => {
                    YuvMatrixCoefficients::Identity
                }
                re_video::decode::YuvMatrixCoefficients::Bt601 => YuvMatrixCoefficients::Bt601,
                re_video::decode::YuvMatrixCoefficients::Bt709 => YuvMatrixCoefficients::Bt709,
            },
            range: match range {
                re_video::decode::YuvRange::Limited => YuvRange::Limited,
                re_video::decode::YuvRange::Full => YuvRange::Full,
            },
        },
    };

    transfer_image_data_to_texture(
        ctx,
        ImageDataDesc {
            label: "video_texture_upload".into(),
            data: std::borrow::Cow::Borrowed(frame.data.as_slice()),
            format,
            width_height: [frame.width, frame.height],
        },
        target_texture,
    )?;

    Ok(format)
}
