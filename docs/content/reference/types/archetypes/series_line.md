---
title: "SeriesLine (deprecated)"
---
<!-- DO NOT EDIT! This file was auto-generated by crates/build/re_types_builder/src/codegen/docs/website.rs -->

⚠️ **Deprecated since 0.23.0**: Use `SeriesLines` instead.
Define the style properties for a line series in a chart.

This archetype only provides styling information and should be logged as static
when possible. The underlying data needs to be logged to the same entity-path using
[`archetypes.Scalars`](https://rerun.io/docs/reference/types/archetypes/scalars).

## Fields
### Optional
* `color`: [`Color`](../components/color.md)
* `width`: [`StrokeWidth`](../components/stroke_width.md)
* `name`: [`Name`](../components/name.md)
* `visible_series`: [`SeriesVisible`](../components/series_visible.md)
* `aggregation_policy`: [`AggregationPolicy`](../components/aggregation_policy.md)


## Can be shown in
* [TimeSeriesView](../views/time_series_view.md)
* [DataframeView](../views/dataframe_view.md)

## API reference links
 * 🌊 [C++ API docs for `SeriesLine`](https://ref.rerun.io/docs/cpp/stable/structrerun_1_1archetypes_1_1SeriesLine.html)
 * 🐍 [Python API docs for `SeriesLine`](https://ref.rerun.io/docs/python/stable/common/archetypes#rerun.archetypes.SeriesLine)
 * 🦀 [Rust API docs for `SeriesLine`](https://docs.rs/rerun/latest/rerun/archetypes/struct.SeriesLine.html)

## Example

### Line series

snippet: archetypes/series_lines_style

<picture data-inline-viewer="snippets/series_lines_style">
  <source media="(max-width: 480px)" srcset="https://static.rerun.io/series_line_style/d2616d98b1e46bdb85849b8669154fdf058e3453/480w.png">
  <source media="(max-width: 768px)" srcset="https://static.rerun.io/series_line_style/d2616d98b1e46bdb85849b8669154fdf058e3453/768w.png">
  <source media="(max-width: 1024px)" srcset="https://static.rerun.io/series_line_style/d2616d98b1e46bdb85849b8669154fdf058e3453/1024w.png">
  <source media="(max-width: 1200px)" srcset="https://static.rerun.io/series_line_style/d2616d98b1e46bdb85849b8669154fdf058e3453/1200w.png">
  <img src="https://static.rerun.io/series_line_style/d2616d98b1e46bdb85849b8669154fdf058e3453/full.png">
</picture>

