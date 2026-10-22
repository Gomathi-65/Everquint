# Water Tank

Given a list of block heights, compute how many units of water are trapped between the blocks and draw the result as an SVG. Built with vanilla JavaScript, HTML and CSS only.

## How to run

No build step and no dependencies.

1. Open `index.html` in a browser (double click it, or use a static server such as the VS Code Live Server extension).
2. Type heights separated by commas, for example `0,4,0,0,0,6,0,6,4,0`, and press **Calculate**.

The example above gives **18 units** of water.

```

## Input rules

- Whole numbers of 0 or more, separated by commas. Square brackets are optional.
- Anything else (letters, negatives, decimals, empty input) shows an error message instead of crashing.

## Approach

The water above a column is `min(tallest block to its left, tallest block to its right) - its own height`, never below zero. Two passes build the left and right maximums, then one pass sums the water. Time O(n), extra space O(n).

## Rendering

The chart is built as an SVG with a grid of square cells: yellow for blocks, blue for water, empty cells as a light grid. The SVG scales to the container width and scrolls horizontally on very narrow screens. Colours are CSS variables with a dark mode that follows the system setting.

## Files

- `index.html` - page structure and form.
- `style.css` - layout, colours and dark mode.
- `script.js` - parsing, water calculation and SVG rendering.
- `test.js` - Node tests for the water calculation (includes the example giving 18).
