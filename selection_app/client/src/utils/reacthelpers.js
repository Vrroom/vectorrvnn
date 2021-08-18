/**
 * @file Helper functions for React
 *
 * @author Sumit Chaturvedi
 */
import { camelCase } from "lodash";

/**
 * Convert all properties to camel case.
 *
 * SVG attributes usually have an hyphen in the
 * middle. This is not allowed when we use them 
 * as React components.
 *
 * @param   {Object}  node - A node in the SVG parse tree
 *
 * @return  {Object}  Node with property names changed
 * to camel case.
 */
function propertiesToCamelCase(node) {
  let { properties, ...rest } = node;
  if (!properties) {
    return rest;
  }
  let newKeys = Object.keys(properties).map(camelCase);
  let oldValues = Object.values(properties);
  let newProperties = {};
  for (let i = 0; i < oldValues.length; i++) {
    const key = newKeys[i];
    const val = oldValues[i];
    newProperties[key] = val;
  }
  return { ...rest, properties: newProperties };
}

export default propertiesToCamelCase;
