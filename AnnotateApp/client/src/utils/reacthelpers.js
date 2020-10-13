import { camelCase } from "lodash";

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
