import { svgTags } from "./svgtags";
import { parse } from "svg-parser";
import propertiesToCamelCase from "./reacthelpers";

function getWidthHeight(props) {
  let height, width;
  if (props.height) {
    if (typeof props.height === "string") {
      height = props.height.replace(/\D/g, "");
    } else {
      height = props.height;
    }
  }
  if (props.width) {
    if (typeof props.width === "string") {
      width = props.width.replace(/\D/g, "");
    } else {
      width = props.width;
    }
  }
  if (!props.height && !props.width) {
    const vb = props.viewBox.split(/ |,|, /);
    width = parseFloat(vb[2]) - parseFloat(vb[0]);
    height = parseFloat(vb[3]) - parseFloat(vb[1]);
  }
  return { width, height };
}

function findBBox(path) {
  const svgElement = document.createElementNS(
    "http://www.w3.org/2000/svg",
    "svg"
  );
  svgElement.setAttribute("id", "temp-svg");
  document.body.appendChild(svgElement);
  const pathElement = document.createElementNS(
    "http://www.w3.org/2000/svg",
    path.tagName
  );
  for (const p in path.properties) {
    pathElement.setAttribute(p, path.properties[p]);
  }
  svgElement.appendChild(pathElement);
  const bbox = svgElement.lastChild.getBBox();
  svgElement.removeChild(pathElement);
  svgElement.remove();
  const {x, y, height, width} = bbox;
  return {x, y, height, width};
}

function flattenTree(tree) {
  let stack = [];
  let elements = [];
  const helper = node => {
    stack.push(node.properties);
    const properties = stack.reduceRight(function(a, b) {
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

function degenerateBBox(bbox) {
  return bbox.width === 0 && bbox.height === 0;
}

function preprocessSVG(svgString) {
  const parseTree = parse(svgString).children[0];
  let { svg, paths } = flattenTree(parseTree);
  let bboxes = paths.map(path => findBBox(path));
  paths = paths.filter((_, i) => !degenerateBBox(bboxes[i]));
  bboxes = bboxes.filter(b => !degenerateBBox(b));
  svg = propertiesToCamelCase(svg);
  paths = paths.map(propertiesToCamelCase);
  return { svg, paths, bboxes };
}

function coveringBBox(bboxes) {
  const x = Math.min(...bboxes.map(b => b.x));
  const y = Math.min(...bboxes.map(b => b.y));
  const maxX = Math.max(...bboxes.map(b => b.x + b.width));
  const maxY = Math.max(...bboxes.map(b => b.y + b.height));
  const height = maxY - y;
  const width = maxX - x;
  return { x, y, height, width };
}

function boxCenter(box) {
  const cx = box.x + box.width / 2;
  const cy = box.y + box.height / 2;
  return { cx, cy };
}

function distance(a, b) {
  const d = { x: a.x - b.x, y: a.y - b.y };
  return Math.sqrt(d.x * d.x + d.y * d.y);
}

function convertCoordinates(elementId, x, y) {
  const ctm = document.getElementById(elementId).getScreenCTM();
  return { x: (x - ctm.e) / ctm.a, y: (y - ctm.f) / ctm.d };
}

export {
  getWidthHeight,
  findBBox,
  flattenTree,
  preprocessSVG,
  coveringBBox,
  boxCenter,
  distance,
  convertCoordinates
};
