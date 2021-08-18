/**
 * @file Functions related to lists.
 *
 * @author Sumit Chaturvedi
 */

import { uniqBy } from "lodash";
import { intersection } from "underscore";

/**
 * Return list of suggestions without duplicate elements.
 *
 * A list of suggestions is a list of tuples as follows:
 * [[0, 1], [1, 0], [0, 5] ... ]
 * The order of the tuples doesn't matter. This function
 * removes duplicates taking into account the order.
 *
 * @param   {Array}   list - List of suggestions.
 *
 * @returns {Array}   List without repetitions.
 */
function uniqTwoTuples(list) {
  const cantorPairingFunction = s => {
    const tuple = s.indices;
    const a = Math.min(tuple[0], tuple[1]);
    const b = Math.max(tuple[0], tuple[1]);
    return 0.5 * (a + b) * (a + b + 1) + b;
  };
  return uniqBy(list, cantorPairingFunction);
}

/*
 * Check if two lists are disjoint.
 *
 * @param   {Array}   list1
 * @param   {Array}   list2
 *
 * @returns {Boolean} Whether they are disjoint or not.
 */
function disjoint(list1, list2) {
  return intersection(list1, list2).length === 0;
}

/*
 * Check if list1 is subset of list2.
 *
 * @param   {Array}   list1
 * @param   {Array}   list2
 *
 * @returns {Boolean} Whether list1 is a subset of list2.
 */
function isSubset(list1, list2) {
  if (list1.length > list2.length) {
    return false;
  }
  return intersection(list1, list2).length === list1.length;
}

/*
 * Check if two lists are same.
 *
 * @param   {Array}   list1
 * @param   {Array}   list2
 *
 * @returns {Boolean} Whether they are disjoint or not.
 */
function identical(list1, list2) {
  if (list1.length !== list2.length) {
    return false;
  }
  return intersection(list1, list2).length === list1.length;
}

function cyclicSlice(list, start, nElements) {
  if (start + nElements > list.length) {
    const rem = start + nElements - list.length;
    const part1 = list.slice(start, list.length);
    const part2 = cyclicSlice(list, 0, rem);
    return part1.concat(part2);
  } else {
    return list.slice(start, start + nElements);
  }
}

export { uniqTwoTuples, disjoint, identical, isSubset, cyclicSlice };
