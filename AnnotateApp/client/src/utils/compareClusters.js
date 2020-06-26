import { intersection, uniq } from "underscore";

function root(tree) {
  for (let i = 0; i < tree.nodes.length; i++) {
    if (!tree.nodes[i].parent) return i;
  }
}

function getRandomInt(max) {
  return Math.floor(Math.random() * Math.floor(max));
}

function selector(a, b) {
  if (a.level < b.level) {
    return a;
  } else if (a.level > b.level) {
    return b;
  } else if (a.ids.length > b.ids.length) {
    return a;
  } else {
    return b;
  }
}

function split(tree, best) {
  let level, arr;
  if (best.ids.length > 1) {
    arr = [...best.ids];
    level = best.level;
  } else {
    const id = best.ids[0];
    arr = [...tree.nodes[id].children];
    level = best.level + 1;
  }
  const index = getRandomInt(arr.length);
  const left = { level, ids: arr.splice(index, 1) };
  const right = { level, ids: arr };
  return { left, right };
}

function treeKCut(tree, k) {
  const r = root(tree);
  const candidates = [{ level: 0, ids: [r] }];
  const leaves = [];
  while (candidates.concat(leaves).length < k) {
    const best = candidates.reduce(selector);
    candidates.splice(candidates.indexOf(best), 1);
    const { left, right } = split(tree, best);
    if (left.ids.length > 1 || tree.nodes[left.ids[0]].children.length > 0) {
      candidates.push(left);
    } else {
      leaves.push(left);
    }
    if (right.ids.length > 1 || tree.nodes[right.ids[0]].children.length > 0) {
      candidates.push(right);
    } else {
      leaves.push(right);
    }
  }
  const cuts =  candidates.concat(leaves).map(candidate =>
    candidate.ids.map(id => tree.nodes[id].paths).flat()
  );
  return cuts;
}

function union(list) {
  return uniq(list.reduce((a, b) => a.concat(b)));
}

function zeros(k) {
  return Array.from(Array(k), () => Array.from(Array(k), () => 0));
}

function sum(M, axis) {
  if (axis === 0) {
    const col = Array.from(Array(M[0].length), () => 0);
    for (let i = 0; i < M.length; i++) {
      for (let j = 0; j < M[i].length; j++) {
        col[j] += M[i][j];
      }
    }
    return col;
  } else if (axis === 1) {
    const row = Array.from(Array(M.length), () => 0);
    for (let i = 0; i < M.length; i++) {
      for (let j = 0; j < M[i].length; j++) {
        row[i] += M[i][j];
      }
    }
    return row;
  } else {
    let total = 0;
    for (let i = 0; i < M.length; i++) {
      for (let j = 0; j < M[i].length; j++) {
        total += M[i][j];
      }
    }
    return total;
  }
}

function mul(a, b) {
  if (typeof a === "number") {
    return a * b;
  }
  return Array.from(Array(a.length), (_, i) => {
    return mul(a[i], b[i]);
  });
}

function compareClusters(t1, t2) {
  const n = t1.nodes[root(t1)].paths.length;
  const bs = [];
  const es = [];
  for (let k = 2; k < n; k++) {
    const cuts1 = treeKCut(t1, k);
    const cuts2 = treeKCut(t2, k);
    const M = zeros(k);
    for (let i = 0; i < cuts1.length; i++) {
      for (let j = 0; j < cuts2.length; j++) {
        M[i][j] = intersection(cuts1[i], cuts2[j]).length;
      }
    }
    const tk = sum(mul(M, M)) - n;
    const mi = sum(M, 0);
    const mj = sum(M, 1);
    const pk = mul(mi, mi).reduce((a, b) => a + b) - n;
    const qk = mul(mj, mj).reduce((a, b) => a + b) - n;
    const bk = tk / Math.sqrt(pk * qk);
    const ek = Math.sqrt(pk * qk) / (n * (n - 1));
    bs.push(bk);
    es.push(ek);
  }
  return { bs, es };
}

export { compareClusters };
