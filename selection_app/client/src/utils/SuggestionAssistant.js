import { uniqTwoTuples, identical, isSubset } from "./listOps";
import { cloneDeep, range } from "lodash";

class SuggestionAssistant {
  /*
   * Construct a SuggestionAssistant object.
   *
   * @param   {Controller}   boss
   */
  constructor(boss) {
    this.boss = boss;
    this.suggestions = [];
    this.maxSuggestions = 3;
  }

  /*
   * Empty local suggestion list.
   */
  emptySuggestionPool = () => {
    this.suggestions = [];
  };

  /*
   * Calculate suggestion priority.
   *
   * @param   {Object}  suggestion - The indices
   * of the nodes, their utility and number of times
   * they have been shown to the annotator.
   *
   * @returns {Number}  The priority. Numerically
   * lesser priority is high priority.
   */
  suggestionScore = suggestion => {
    const { utility, timesShown } = suggestion;
    return (1 - utility) * timesShown;
  };

  /*
   * Check whether there is an identical suggestion in the list.
   *
   * @param   {Array}   suggestionList - List of suggestions.
   * @param   {Object}  suggestion - Suggestion to be searched for.
   *
   * @returns {Boolean} Whether there is even one matching suggestion.
   */
  checkIdenticalSuggestion = (suggestionList, suggestion) => {
    return suggestionList.some(s => identical(suggestion.indices, s.indices));
  };

  /*
   * Calculate the number of trees in the forest.
   *
   * Simple way to do this is to calculate the number of
   * root nodes.
   */
  nTrees = () => {
    const { graph } = this.boss.state;
    return graph.nodes
      .map(n => (typeof n.parent === "undefined" ? 1 : 0))
      .reduce((a, b) => a + b, 0);
  };

  /*
   * Update suggestions based on clock ticks.
   *
   * Every 5 seconds, if required, new suggestions
   * needs to be fetched from the server.
   *
   * If some suggestion has expired, that is that
   * it has been shown to the annotator for
   * 10 seconds and hasn't been accepted, then
   * it needs to be replaced.
   *
   * The choice of replacement depends
   * on how many times the suggestions have
   * been shown and what the suggestions utility
   * was, as deemed by the grouping model.
   *
   * If needed, the expired suggestions are removed
   * and new suggestions are shown to the user
   * and the Controller's suggestion state is
   * set using setState.
   */
  timeCheck = () => {
    const { time } = this.boss.state;
    this.boss.setState({ time: time + 1 });
    // Get new suggestion every 5 seconds.
    if (time % 50 === 0 && this.nTrees() > 1) {
      this.fetchSuggestion();
    }
    // Check which suggestions in the boss pool of
    // suggestions have expired. Remove them
    // from the pool. And update their timesShown
    // count.
    let { suggestions } = this.boss.state;
    const expiredSuggestions = suggestions.filter(
      s => time - s.startTime >= 100
    );
    // If no expired suggestion exists, return.
    const noExpiredSuggestion = expiredSuggestions.length === 0;
    const maxSuggestionsGiven = suggestions.length === this.maxSuggestions;
    if (noExpiredSuggestion && maxSuggestionsGiven) {
      return;
    }
    // Increment timesShown for expired suggestions in the
    // main suggestion pool.
    for (let i = 0; i < this.suggestions.length; i++) {
      const suggestion = this.suggestions[i];
      if (this.checkIdenticalSuggestion(expiredSuggestions, suggestion)) {
        suggestion.timesShown += 1;
      }
    }
    // Update the suggestions. First filter
    // the expired suggestions. Then, find the
    // suggestions with lowest scores and fill
    // the suggestion list with them.
    suggestions = suggestions.filter(s => time - s.startTime < 100);
    let poolCopy = cloneDeep(this.suggestions);
    poolCopy.sort((a, b) => this.suggestionScore(a) - this.suggestionScore(b));
    poolCopy = poolCopy.filter(
      s => !this.checkIdenticalSuggestion(suggestions, s)
    );
    const leftOver = this.maxSuggestions - suggestions.length;
    const newSuggestions = poolCopy.slice(0, leftOver);
    suggestions = suggestions.concat(
      newSuggestions.map(s => {
        return { ...s, startTime: time };
      })
    );
    this.boss.setState({ suggestions: suggestions });
  };

  /*
   * Update the graph and suggestions in this.boss.
   *
   * The suggestions depend on the state of the graph.
   * Hence to avoid errors, they need to be updated
   * simultaneously.
   *
   * Here, we make the suggestions consistent with the
   * graph and update the suggestions as well as the
   * graph both within this class as well in the
   * Controller class using the setState method.
   *
   * @param   {Object}  graph - The graph object.
   */
  graphCheck = graph => {
    this.suggestions = this.consistentSuggestions(graph, this.suggestions);
    const suggestions = this.consistentSuggestions(
      graph,
      this.boss.state.suggestions
    );
    this.boss.setState({ graph, suggestions, pointId: graph.nodes.length - 1 });
  };

  /*
   * Make the current suggestion state consistent with the graph.
   *
   * Basically all the candidates in the suggestion should be
   * root nodes of their subtrees. Also, they should exist in
   * the graph.
   *
   * @param   {Object}  graph - The graph.
   * @param   {Array}   suggestions - The suggestions.
   */
  consistentSuggestions = (graph, suggestions) => {
    const { nodes } = graph;
    const n = nodes.length;
    suggestions = suggestions.filter(s => isSubset(s.indices, range(n)));
    suggestions = suggestions.filter(s =>
      s.indices.every(i => typeof nodes[i].parent === "undefined")
    );
    return suggestions;
  };

  /*
   * Cycle through node ids to fetch suggestions.
   *
   * Every time a new suggestion is required
   * to be calculated, increment the nodeId to
   * point to a fresh root node.
   */
  nextSuggestionNodeId = () => {
    const { nodes } = this.boss.state.graph;
    const n = nodes.length;
    if (n === 0) {
      return;
    }
    this.nodeId =
      (typeof this.nodeId === "undefined" ? 0 : this.nodeId + 1) % n;
    while (typeof nodes[this.nodeId].parent !== "undefined") {
      this.nodeId = (this.nodeId + 1) % n;
    }
  };

  /*
   * Add suggestion to suggestion pool.
   *
   * @param   {Array}   suggestion - An array containing just
   * zero or one suggestions. Because a suggestion might not
   * be available and Optionals don't work right now, this
   * seemed to be the cleanest way of getting things done.
   */
  updateSuggestionList = suggestion => {
    let suggestions = cloneDeep(this.suggestions);
    suggestions = suggestions.concat(suggestion);
    suggestions = uniqTwoTuples(suggestions);
    suggestions = this.consistentSuggestions(
      this.boss.state.graph,
      suggestions
    );
    this.suggestions = suggestions;
  };

  /*
   * Fetch suggestion from grouping algorithm
   * running on the server.
   */
  fetchSuggestion = () => {
    // TODO : Change the lower bound.
    // this.nextSuggestionNodeId();
    // fetch("/suggestion", {
    //   method: "post",
    //   headers: {
    //     Accept: "application/json, text/plain, */*",
    //     "Content-Type": "application/json"
    //   },
    //   body: JSON.stringify({
    //     id: this.boss.state.id,
    //     graph: this.boss.state.graph,
    //     nodeId: this.nodeId,
    //     lowerBound: -1000
    //   })
    // })
    //   .then(res => res.json())
    //   .then(item => {
    //     if (item.id === this.boss.state.id) {
    //       this.updateSuggestionList(item.suggestion);
    //     }
    //   });
  };
}

export default SuggestionAssistant;
