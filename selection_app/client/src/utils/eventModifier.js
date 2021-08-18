/**
 * @file Functions which modify the event handler functions.
 *
 * @author Sumit Chaturvedi
 */

/**
 * Invoke stopPropagation and handle the event
 * as before.
 *
 * @param   {function}  eventHandler
 *
 * @return  {function}  An event handler which stops the
 * propagation of the event to the parent and
 * then invokes the eventHandler.
 */
function addStopPropagation(eventHandler) {
  const newEventHandler = event => {
    event.stopPropagation();
    eventHandler(event);
  };
  return newEventHandler;
}

export default addStopPropagation;
