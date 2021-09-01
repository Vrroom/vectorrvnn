/**
 * @file Functions related the SVGs.
 *
 * @author Sumit Chaturvedi
 */
import { svgTags } from "./svgtags";
import { parse } from "svg-parser";
import propertiesToCamelCase from "./reacthelpers";
import { Shape, Path, extend, SVG } from "@svgdotjs/svg.js";
import { cloneDeep } from "lodash";
import * as d3 from "d3-color";

extend(Shape, {
  // Convert element to path
  toPath(replace = false) {
    var d;

    switch (this.type) {
      case "rect": {
        let { width: w, height: h, rx, ry, x, y } = this.attr(["width", "height", "rx", "ry", "x", "y"]);

        // normalise radius values, just like the original does it (or should do)
        if (rx < 0) rx = 0;
        if (ry < 0) ry = 0;
        rx = rx || ry;
        ry = ry || rx;
        if (rx > w / 2) rx = w / 2;
        if (ry > h / 2) ry = h / 2;

        if (rx && ry) {
          // if there are round corners

          d = [
            ["M", rx + x, y],
            ["h", w - 2 * rx],
            ["a", rx, ry, 0, 0, 1, rx, ry],
            ["v", h - 2 * ry],
            ["a", rx, ry, 0, 0, 1, -rx, ry],
            ["h", -w + 2 * rx],
            ["a", rx, ry, 0, 0, 1, -rx, -ry],
            ["v", -h + 2 * ry],
            ["a", rx, ry, 0, 0, 1, rx, -ry],
            ["z"],
          ];
        } else {
          // no round corners, no need to draw arcs
          d = [["M", x, y], ["h", w], ["v", h], ["h", -w], ["v", -h], ["z"]];
        }

        break;
      }
      case "circle":
      case "ellipse": {
        let rx = this.rx();
        let ry = this.ry();
        let { cx, cy } = this.attr(["cx", "cy"]);

        d = [["M", cx - rx, cy], ["A", rx, ry, 0, 0, 0, cx + rx, cy], ["A", rx, ry, 0, 0, 0, cx - rx, cy], ["z"]];

        break;
      }
      case "polygon":
      case "polyline":
      case "line":
        d = this.array().map(function (arr) {
          return ["L"].concat(arr);
        });

        d[0][0] = "M";

        if (this.type === "polygon") {
          d.push("Z");
        }

        break;
      case "path":
        d = this.array();
        break;
      default:
        throw new Error("SVG toPath got unsupported type " + this.type, this);
    }

    const path = new Path().plot(d);

    return path;
  },
});

function initialCanvasTransform(graphic, pathIdx) {
  const { bboxes } = graphic;
  const relevantBoxes = pathIdx.map((i) => bboxes[i]);
  const box = coveringBBox(relevantBoxes);
  const center = boxCenter(box);
  const t1 = "translate(50 50)";
  const s = "scale(1 1)";
  const t2 = `translate(-${center.cx} -${center.cy})`;
  return [t1, s, t2];
}

function transformBox(transformList, box) {
  for (let i = 0; i < transformList.length; i++) {
    const transform = parseTransform(transformList[i]);
    if (transform.type === "scale") {
      box.width *= transform.x;
      box.height *= transform.y;
    } else if (transform.type === "translate") {
      box.x += transform.x;
      box.y += transform.y;
    }
  }
  return box;
}

function parseTransform(transform) {
  const t = cloneDeep(transform);
  const open = t.indexOf("(");
  const close = t.indexOf(")");
  const cut = t.slice(open + 1, close);
  const space = cut.indexOf(" ");
  const x = parseFloat(cut.slice(0, space));
  const y = parseFloat(cut.slice(space + 1, cut.length));
  return { type: t.slice(0, open), x, y };
}

function point(id) {
  return SVG(`#${id}`).toPath().pointAt(0);
}

/**
 * Calculate the height or width from the SVG
 * document property.
 *
 * It is possible that the units are in px. This
 * convenience function extracts all the digits.
 *
 * @param   {Object}  dim - SVG document property for either
 * width or height.
 *
 * @return  {Number}  Extracted width or height value.
 */
function extractDims(dim) {
  if (typeof dim === "string") {
    return dim.replace(/\D/g, "");
  } else {
    return dim;
  }
}

/**
 * Parse the SVG document's viewbox.
 *
 * @param   {string}  vb - Viewbox of the SVG.
 *
 * @return  {Object}  List of 4 numbers specifying the
 * viewbox.
 */
function extractViewBox(vb) {
  return vb.split(/ |,|, /).map(parseFloat);
}

/**
 * Calculate the width and height of the SVG document.
 *
 * @param   {Object}  props - SVG document properties.
 *
 * @return  {Object}  { width, height } as numbers.
 */
function getWidthHeight(props) {
  let height, width;
  if (props.height) {
    height = extractDims(props.height);
  }
  if (props.width) {
    width = extractDims(props.width);
  }
  if (!props.height && !props.width) {
    const vb = extractViewBox(props.viewBox);
    width = vb[2] - vb[0];
    height = vb[3] - vb[1];
  }
  return { width, height };
}

/**
 * Modify the graphic such that the SVG's viewbox
 * is a square and the original graphic window is
 * placed in the middle of the square.
 *
 * Since we might have arbitrary vector graphics
 * in our database, it makes sense to normalize them
 * by making them fit in a square so that they are
 * rendered properly on the browser.
 *
 * @param   {Object}  graphic - The SVG document representation.
 *
 * @return  {Object}  Graphic object whose height and width have
 * been made equal.
 */
function normalizeGraphic(graphic) {
  let { svg, paths } = graphic;
  if (svg.properties.viewBox) {
    const vb = extractViewBox(svg.properties.viewBox);
    if (vb[2] < vb[3]) {
      svg.properties.viewBox = `${vb[0] - (vb[3] - vb[2]) / 2} ${vb[1]} ${vb[3]} ${vb[3]}`;
    } else {
      svg.properties.viewBox = `${vb[0]} ${vb[1] - (vb[2] - vb[3]) / 2} ${vb[2]} ${vb[2]}`;
    }
    svg.properties.height = Math.max(vb[2], vb[3]);
    svg.properties.width = Math.max(vb[2], vb[3]);
  } else if (svg.properties.height && svg.properties.width) {
    const h = extractDims(svg.properties.height);
    const w = extractDims(svg.properties.width);
    const max = Math.max(h, w);
    svg.properties.height = max;
    svg.properties.width = max;
    svg.properties.viewBox = `0 0 ${max} ${max}`;
  }
  return { svg, paths };
}

function fitGraphicIn100By100Box(graphic) {
  let { svg, paths } = graphic;
  const { width, height } = getWidthHeight(svg.properties);
  const d = Math.max(width, height);
  const transform = `scale(${100 / d} ${100 / d})`;
  svg.properties.height = 100;
  svg.properties.width = 100;
  const vb = extractViewBox(svg.properties.viewBox);
  vb[0] *= 100 / d;
  vb[1] *= 100 / d;
  vb[2] = 100;
  vb[3] = 100;
  svg.properties.viewBox = vb.join(" ");
  paths = paths.map((path) => {
    return { tagName: "g", transform, children: [path] };
  });
  return { svg, paths, scale: 100 / d };
}

/**
 * Find the bounding box of a path.
 *
 * 1. Create a temporary SVG element in browser.
 * 2. Add the path to that SVG element.
 * 3. Use the browser function getBBox to calculate
 *    the bounding box.
 * 4. Clean up the temporary element.
 *
 * @param   {Object}  path - SVG path.
 *
 * @return  {Object}  A bounding box for the path.
 */
function findBBox(group) {
  const path = group.children[0];
  const svgElement = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svgElement.setAttribute("id", "temp-svg");
  svgElement.setAttribute("height", "100");
  svgElement.setAttribute("width", "100");
  document.body.appendChild(svgElement);
  const groupElement = document.createElementNS("http://www.w3.org/2000/svg", group.tagName);
  groupElement.setAttribute("transform", group.transform);
  const pathElement = document.createElementNS("http://www.w3.org/2000/svg", path.tagName);
  for (const p in path.properties) {
    pathElement.setAttribute(p, path.properties[p]);
  }
  groupElement.appendChild(pathElement);
  svgElement.appendChild(groupElement);
  const bbox = svgElement.lastChild.getBBox();
  svgElement.removeChild(groupElement);
  svgElement.remove();
  let { x, y, height, width } = bbox; // bbox is of type SVGRect.
  const scale = parseTransform(group.transform);
  x *= scale.x;
  y *= scale.y;
  height *= scale.x;
  width *= scale.y;
  return { x, y, height, width };
}

/**
 * Extract the paths from SVG parse tree.
 *
 * While flattening the parse tree, it is
 * ensured that the inherited properties
 * are set correctly.
 *
 * @param   {Object}  tree - SVG parse tree.
 *
 * @return  {Object}  The object consists of an svg and a paths
 * property. The svg property stores information about the whole
 * document. The paths property is a list of paths.
 */
function flattenTree(tree) {
  let stack = [];
  let elements = [];
  const helper = (node) => {
    stack.push(node.properties);
    const properties = stack.reduceRight(function (a, b) {
      return { ...b, ...a };
    });
    let { children, ...rest } = node;
    if (svgTags.includes(rest.tagName) || rest.tagName === "svg") {
      rest.properties = properties;
      elements.push(rest);
    }
    children.forEach(helper);
    stack.pop();
  };
  helper(tree);
  return { svg: elements[0], paths: elements.slice(1) };
}

/**
 * Check whether bounding box is simply a point.
 *
 * Many times, the paths in the document will
 * have no geometry. Such paths are useless
 * for our purpose.
 *
 * We detect them by determining whether they
 * occupy 0 area or not.
 *
 * @param   {Object}  bbox - Bounding box.
 *
 * @return  {boolean} True if bounding box is simply a point.
 */
function degenerateBBox(bbox) {
  return bbox.width === 0 && bbox.height === 0;
}

/**
 * Process an SVG string.
 *
 * This function performs a bunch of operations
 * on the SVG string fetched from the database.
 *
 * 1. The string is parsed.
 * 2. The parse tree is flattened. We are only concerned
 *    with the paths in the SVG. Flattening the parse tree
 *    makes it more convenient to index the paths.
 * 3. Bounding boxes are computed for the paths.
 * 4. The SVG properties are converted to camel case
 *    so that they match React's requirements for
 *    jsx elements.
 * 5. Finally, the SVG is normalized to fit into a
 *    square viewbox.
 *
 * @param   {string}  svgString - SVG document as a string.
 *
 * @return  {object}  A graphic object having the SVG, it's
 * paths and their bounding boxes as properties.
 */
function preprocessSVG(svgString) {
  const parseTree = parse(svgString).children[0];
  let { svg, paths, scale } = fitGraphicIn100By100Box(normalizeGraphic(flattenTree(parseTree)));
  let bboxes = paths.map((path) => findBBox(path));
  paths = paths.filter((_, i) => !degenerateBBox(bboxes[i]));
  bboxes = bboxes.filter((b) => !degenerateBBox(b));
  svg = propertiesToCamelCase(svg);
  paths = paths.map(propertiesToCamelCase);
  return { svg, paths, bboxes, scale };
}

/**
 * Calculate a bounding box which covers all the input
 * bounding boxes.
 *
 * This function is used when we want to compute the
 * bounding box for a set of paths which have been
 * grouped by the user.
 *
 * @param   {Object}  bboxes - List of bounding boxes
 *
 * @return  {Object}  The smallest bounding box which
 * covers all bboxes.
 */
function coveringBBox(bboxes) {
  const x = Math.min(...bboxes.map((b) => b.x));
  const y = Math.min(...bboxes.map((b) => b.y));
  const maxX = Math.max(...bboxes.map((b) => b.x + b.width));
  const maxY = Math.max(...bboxes.map((b) => b.y + b.height));
  const height = maxY - y;
  const width = maxX - x;
  return { x, y, height, width };
}

function fourCorners(bbox) {
  const { x, y, width, height } = bbox;
  const corners = [
    { x, y },
    { x: x + width, y },
    { x: x + width, y: y + height },
    { x, y: y + height },
  ];
  return corners;
}

/**
 * Calculate the center of a box.
 *
 * @param   {Object}  box - Must have x, y, width and height
 * properties.
 *
 * @return  {Object}  The center point as { cx, cy }
 */
function boxCenter(box) {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  return { cx, cy };
}

/**
 * Calculate euclidean distance between two points in 2D.
 *
 * @param   {Object}  a - Point a given as { x, y }.
 * @param   {Object}  b - Point b.
 *
 * @return  {Number}  Euclidean distance.
 */
function distance(a, b) {
  const d = subtract(a, b);
  return Math.sqrt(d.x * d.x + d.y * d.y);
}

function subtract(a, b) {
  const d = { x: a.x - b.x, y: a.y - b.y };
  return d;
}

/**
 * Convert coordinates from the document's
 * to the element's whose id is supplied.
 *
 * Used to convert event coordinates on a
 * particular element.
 *
 * @param   {string}  elementId - Id of element
 * @param   {Number}  x - x coordinate of the point.
 * @param   {Number}  y - y coordinate of the point.
 *
 * @return  {Object}  The transformed point with x and y
 * properties.
 */
function convertCoordinates(elementId, x, y) {
  const ctm = document.getElementById(elementId).getScreenCTM();
  return { x: (x - ctm.e) / ctm.a, y: (y - ctm.f) / ctm.d };
}

function getAttribute (properties, attr) { 
  if (typeof properties[attr] === "undefined") {
    if (typeof properties.style === "undefined") return "none";
    else if (typeof properties.style[attr] === "undefined") return "none";
    else return properties.style[attr];
  }
  return properties[attr];
}

function setAttribute (properties, attr, newVal) {
  const prop = cloneDeep(properties);
  if (typeof properties[attr] === "undefined") {
    if (typeof properties.style === "undefined") return prop;
    else if (typeof properties.style[attr] === "undefined") return prop;
    else {
      prop.style[attr] = newVal;
      return prop;
    }
  }
  prop[attr] = newVal;
  return prop;
}

function isAttrNotNone(properties, attr) {
  return getAttribute(properties, attr).toLowerCase() !== "none";
}

function rgb2gray (color) {
  const avg = (color.r + color.g + color.b) / 3;
  color.r = color.g = color.b = avg;
  return color + "";
}

function targetGraphic (graphic, target) {
  const graphic_ = cloneDeep(graphic);
  for (let i = 0; i < graphic_.paths.length; i++) {
    let group = graphic_.paths[i];
    let child = group.children[0];
    if (target.includes(i)) {
      child.properties = setAttribute(child.properties, "stroke", "red");
      child.properties = setAttribute(child.properties, "fill"  , "red");
    } else {
      if (isAttrNotNone(child.properties, "stroke")) { 
        const strokeColor = d3.color(getAttribute(child.properties, "stroke"))
        const grayStroke  = rgb2gray(strokeColor);
        child.properties  = setAttribute(child.properties, "stroke", grayStroke);
      }
      if (isAttrNotNone(child.properties, "fill")) { 
        const fillColor   = d3.color(getAttribute(child.properties, "fill"))
        const grayFill    = rgb2gray(fillColor);
        child.properties  = setAttribute(child.properties, "fill", grayFill);
      }
    }
  }
  return graphic_
}

export {
  point,
  getWidthHeight,
  preprocessSVG,
  coveringBBox,
  boxCenter,
  distance,
  subtract,
  convertCoordinates,
  initialCanvasTransform,
  fourCorners,
  parseTransform,
  transformBox,
  isAttrNotNone,
  setAttribute,
  targetGraphic,
  extractViewBox,
};
