import re

def reverse_f_string(s, fstring_pattern, var_types, scope=None):
    """ 
    Extracts variables from a string based on an f-string-like pattern. Optionally updates a provided scope with these variables.

    Parameters
    ----------
    s : str
        The string to be processed, which is expected to match the format defined by `fstring_pattern`.
    
    fstring_pattern : str
        The f-string-like pattern used to parse `s`. Variables in the pattern should be enclosed in curly braces, e.g., '{variable}'.

    var_types : type or list of types
        The type or a list of types to which the extracted string values should be converted. If a list is provided, it should be in the 
        same order as the variables in `fstring_pattern`.

    scope : dict, optional
        The scope in which extracted variables should be updated. If provided, this function will update the scope with the extracted variables.
        If None (default), no scope is updated, and a dictionary of extracted variables is returned.

    Returns
    -------
    dict
        A dictionary containing the extracted variables and their values, converted to the specified types.

    Raises
    ------
    ValueError
        If the string `s` does not match the `fstring_pattern`, if the number of types provided does not match the number of variables,
        or if a type conversion fails.

    Example
    -------
    >>> values = reverse_f_string('epoch=0_step=4.ckpt', 'epoch={epoch}_step={step}.ckpt', [int, int], locals())
    >>> epoch, step = values['epoch'], values['step']
    >>> print(epoch, step)

    Notes
    -----
    - The function assumes that `fstring_pattern` contains simple variable placeholders and does not support complex expressions or format specifications.
    - When `scope` is provided, it must be a mutable dictionary-like object (e.g., the result of calling `locals()` or `globals()` in the calling scope).
    - The `var_types` parameter should either be a single type (if there's only one variable) or a list of types corresponding to each variable in order.
    """
    # Extract variable names from f-string-like pattern
    var_names = re.findall(r'\{(.*?)\}', fstring_pattern)

    # Validate and construct the regex pattern
    regex_pattern = fstring_pattern
    for var in var_names:
        regex_pattern = regex_pattern.replace(f"{{{var}}}", r"(.+?)")

    # Match against the string
    match = re.match(regex_pattern, s)
    if not match:
        raise ValueError("No match found")

    # Ensure each variable name has exactly one match
    if len(match.groups()) != len(var_names):
        raise ValueError("Number of matches and variables do not correspond")

    # Convert parsed strings to specified types and return as a dict
    values = {}
    for i, var in enumerate(var_names):
        try:
            # Apply the type conversion
            var_type = var_types[i] if isinstance(var_types, list) else var_types
            value = var_type(match.group(i + 1))
            values[var] = value
        except ValueError as e:
            raise ValueError(f"Conversion error for variable '{var}': {e}")

    if scope is not None:
        scope.update(values)
    return values
