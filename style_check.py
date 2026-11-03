"""Check the parts of the Python style guide that ruff cannot express.

Ruff enforces most of the guide (see ruff.toml). This script enforces the rest,
with the standard library's ``ast``, so that Python is held to the same bar as
the TypeScript ESLint configuration in this repository:

- Naming: functions and methods are ``<subject>[Qualifier]_<verb>[Adverb]``,
  lowerCamelCase halves, exactly one underscore, and a verb, not a
  preposition, after it. Properties are lowerCamelCase nouns. The pattern is
  the one in typescript/eslint.config.mjs.
- Docstrings everywhere, private helpers included (ruff checks public names).
- An ``Args:`` section on every function that takes parameters (ruff checks
  the section's contents only when it exists).
- An ``Example:`` section on every public function, method and class.
- An ``Attributes:`` entry for every public field a class declares.
- A module docstring with ``Key components:``, ``Typical usage:`` and
  ``Dependencies:`` sections, except in test modules.

Key components:
    - Violation: one finding, with its location and the reason.
    - Scope: what encloses a definition, which decides what it must carry.
    - functionName_check: judge one function name.
    - functionDocstring_check: judge one function docstring.
    - classDocstring_check: judge one class docstring.
    - moduleDocstring_check: judge the module docstring.
    - violations_collect: walk one source file and judge everything in it.
    - main: command-line entry point.

Typical usage:
    python style_check.py src/ tests/

Exemptions, all of them names a framework or the language dictates:
    - dunder methods keep their names and need only a docstring, plus
      ``Args:`` for ``__init__``;
    - methods decorated with ``@override`` (``typing`` or
      ``typing_extensions``) keep the base class's name and its documentation;
    - pytest's ``test_*`` functions keep their names and need only a docstring;
    - ``main`` is the one function name without a subject.

Dependencies:
    Standard library only. Exits 0 when every check passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PREPOSITIONS: str = "of|to|with|from|for|by|as|in|on|at"
RPN_PATTERN: re.Pattern[str] = re.compile(
    rf"^(?:main|[a-z][a-zA-Z0-9]*_(?!(?:{PREPOSITIONS})(?:[A-Z0-9]|$))[a-z][a-zA-Z0-9]*)$"
)
PREPOSITION_PATTERN: re.Pattern[str] = re.compile(rf"^(?:{PREPOSITIONS})(?:[A-Z0-9]|$)")
FIELD_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-zA-Z0-9]*$")
SECTION_PATTERN: re.Pattern[str] = re.compile(r"^([A-Z][a-z]+(?: [a-z]+)*):\s*$", re.MULTILINE)
ATTRIBUTE_PATTERN: re.Pattern[str] = re.compile(
    r"^ {4}([A-Za-z_]\w*)(?: \([^)]*\))?:", re.MULTILINE
)
TEST_MODULE_PATTERN: re.Pattern[str] = re.compile(r"^(?:test_.*|.*_test|conftest)\.py$")
MODULE_SECTIONS: tuple[str, ...] = ("Key components", "Typical usage", "Dependencies")
EXAMPLE_SECTIONS: frozenset[str] = frozenset({"Example", "Examples"})
EXEMPT_DECORATORS: frozenset[str] = frozenset({"override"})
PROPERTY_DECORATORS: frozenset[str] = frozenset({"property", "cached_property"})
IMPLICIT_PARAMETERS: frozenset[str] = frozenset({"self", "cls"})
UNDOCUMENTED_FIELDS: frozenset[str] = frozenset({"model_config"})
PYTEST_PREFIX: str = "test_"
EXIT_CLEAN: int = 0
EXIT_VIOLATIONS: int = 1
DESCRIPTION: str = "Check what ruff cannot: RPN naming and docstring structure."
PREPOSITION_REASON: str = (
    "the part after the underscore is a preposition; name the verb (derive, parse, render)"
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


@dataclass(frozen=True)
class Violation:
    """One finding, in a form an editor can jump to.

    Attributes:
        path: Source file the finding is in.
        lineNumber: One-based line of the definition.
        name: The definition's name, or ``<module>``.
        reason: What is wrong, phrased for the person fixing it.

    Example:
        >>> Violation(Path("a.py"), 3, "getUser", "not subject_verb").report
        'a.py:3: getUser: not subject_verb'
    """

    path: Path
    lineNumber: int
    name: str
    reason: str

    @property
    def report(self) -> str:
        """One line in the ``path:line: name: reason`` form."""
        return f"{self.path}:{self.lineNumber}: {self.name}: {self.reason}"


@dataclass(frozen=True)
class Scope:
    """What encloses a definition, which decides what it must carry.

    Attributes:
        isPublic: True when every enclosing class is public and the definition
            is at module or class level, not nested in a function.
        isTestModule: True in a pytest module, where the module docstring needs
            no sections.

    Example:
        >>> Scope(isPublic=True, isTestModule=False).isPublic
        True
    """

    isPublic: bool
    isTestModule: bool


def decoratorNames_get(node: FunctionNode | ast.ClassDef) -> set[str]:
    r"""Collect the final names of a definition's decorators.

    Args:
        node: A function, async function or class definition.

    Returns:
        Decorator names with any module prefix and call removed, so
        ``@typing.override`` and ``@functools.cached_property()`` give
        ``override`` and ``cached_property``. A setter such as
        ``@startTime.setter`` gives ``property``, because it belongs to one.

    Example:
        >>> tree = ast.parse("@typing.override\ndef f(): pass")
        >>> decoratorNames_get(tree.body[0])
        {'override'}
    """
    names: set[str] = set()
    for decorator in node.decorator_list:
        target: ast.expr = decorator.func if isinstance(decorator, ast.Call) else decorator
        match target:
            case ast.Attribute(attr="setter" | "deleter"):
                names.add("property")
            case ast.Attribute(attr=attr):
                names.add(attr)
            case ast.Name(id=identifier):
                names.add(identifier)
    return names


def docstringSections_get(docstring: str) -> set[str]:
    r"""Find the Google-style section headings in a docstring.

    Args:
        docstring: A docstring as ``ast.get_docstring`` returns it, with its
            indentation already removed.

    Returns:
        Heading names without the colon, such as ``{"Args", "Returns"}``.

    Example:
        >>> sorted(docstringSections_get("Sum.\n\nArgs:\n    x: X.\n\nReturns:\n    Y."))
        ['Args', 'Returns']
    """
    return set(SECTION_PATTERN.findall(docstring))


def functionName_check(node: FunctionNode) -> str | None:
    """Judge one function name against the naming rule.

    Args:
        node: A function or async function definition.

    Returns:
        ``None`` if the name passes or is exempt, otherwise the reason it fails.

    Example:
        >>> functionName_check(ast.parse("def answerRows_of(): pass").body[0])
        'the part after the underscore is a preposition; name the verb (derive, parse, render)'
    """
    name: str = node.name
    decorators: set[str] = decoratorNames_get(node)
    isExempt: bool = (
        (name.startswith("__") and name.endswith("__"))
        or bool(decorators & EXEMPT_DECORATORS)
        or name.startswith(PYTEST_PREFIX)
    )
    if isExempt:
        return None
    publicName: str = name.removeprefix("_")
    if decorators & PROPERTY_DECORATORS:
        if FIELD_PATTERN.match(publicName):
            return None
        return "a property is named like a field: a lowerCamelCase noun"
    if RPN_PATTERN.match(publicName):
        return None
    if PREPOSITION_PATTERN.match(publicName.partition("_")[2]):
        return PREPOSITION_REASON
    return "expected <subject>_<verb>, lowerCamelCase halves, one underscore"


def parameterNames_get(node: FunctionNode) -> list[str]:
    """List the parameters a docstring must describe.

    Args:
        node: A function or async function definition.

    Returns:
        Parameter names in signature order, without ``self`` and ``cls``,
        with ``*args`` and ``**kwargs`` included by name.

    Example:
        >>> parameterNames_get(ast.parse("def f(self, a, *b, c, **d): pass").body[0])
        ['a', 'b', 'c', 'd']
    """
    arguments: ast.arguments = node.args
    every: list[ast.arg | None] = [
        *arguments.posonlyargs,
        *arguments.args,
        arguments.vararg,
        *arguments.kwonlyargs,
        arguments.kwarg,
    ]
    return [
        argument.arg
        for argument in every
        if argument is not None and argument.arg not in IMPLICIT_PARAMETERS
    ]


def functionDocstring_check(node: FunctionNode, scope: Scope) -> list[str]:
    r"""Judge one function's docstring against the documentation rules.

    Properties, ``@override`` methods, dunders and pytest tests need only a
    docstring (``__init__`` also needs ``Args:``). Every other function with
    parameters needs ``Args:``; every public one needs ``Example:``. Returns,
    yields and raises are ruff's (``DOC`` rules), so they are not checked here.

    Args:
        node: A function or async function definition.
        scope: What encloses the definition.

    Returns:
        One reason per missing element; empty when the docstring passes.

    Example:
        >>> node = ast.parse("def value_get(count):\n    '''Get.'''").body[0]
        >>> functionDocstring_check(node, Scope(isPublic=True, isTestModule=False))
        ['docstring has no Args: section', 'public function has no Example: section']
    """
    docstring: str | None = ast.get_docstring(node)
    if docstring is None:
        return ["missing docstring"]
    name: str = node.name
    decorators: set[str] = decoratorNames_get(node)
    isDunder: bool = name.startswith("__") and name.endswith("__")
    needsOnlyDocstring: bool = (
        bool(decorators & (PROPERTY_DECORATORS | EXEMPT_DECORATORS))
        or name.startswith(PYTEST_PREFIX)
        or (isDunder and name != "__init__")
    )
    if needsOnlyDocstring:
        return []
    sections: set[str] = docstringSections_get(docstring)
    reasons: list[str] = []
    if parameterNames_get(node) and "Args" not in sections:
        reasons.append("docstring has no Args: section")
    isPublic: bool = scope.isPublic and not name.startswith("_")
    if isPublic and not sections & EXAMPLE_SECTIONS:
        reasons.append("public function has no Example: section")
    return reasons


def fieldNames_get(node: ast.ClassDef) -> list[str]:
    r"""List the public fields a class declares with annotations.

    Args:
        node: A class definition.

    Returns:
        Names of annotated class-level fields, in order, without private
        (``_``-prefixed) names and framework settings such as ``model_config``.

    Example:
        >>> fieldNames_get(ast.parse("class A:\n    x: int\n    _y: int").body[0])
        ['x']
    """
    names: list[str] = []
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name: str = statement.target.id
            if not name.startswith("_") and name not in UNDOCUMENTED_FIELDS:
                names.append(name)
    return names


def attributeNames_get(docstring: str) -> set[str]:
    r"""Find the names documented in a docstring's ``Attributes:`` section.

    Args:
        docstring: A class docstring with its indentation already removed.

    Returns:
        Names that open an entry under ``Attributes:``; empty if there is no
        such section.

    Example:
        >>> attributeNames_get("Sum.\n\nAttributes:\n    x: The x.\n\nExample:\n    y")
        {'x'}
    """
    _, found, rest = docstring.partition("Attributes:\n")
    if not found:
        return set()
    nextSection: re.Match[str] | None = SECTION_PATTERN.search(rest)
    section: str = rest[: nextSection.start()] if nextSection is not None else rest
    return set(ATTRIBUTE_PATTERN.findall(section))


def classDocstring_check(node: ast.ClassDef, scope: Scope) -> list[str]:
    r"""Judge one class's docstring against the documentation rules.

    Args:
        node: A class definition.
        scope: What encloses the definition.

    Returns:
        One reason per missing element; empty when the docstring passes.

    Example:
        >>> node = ast.parse("class Point:\n    '''A point.'''\n    x: int").body[0]
        >>> classDocstring_check(node, Scope(isPublic=True, isTestModule=False))
        ['field x is not under Attributes:', 'public class has no Example: section']
    """
    docstring: str | None = ast.get_docstring(node)
    if docstring is None:
        return ["missing docstring"]
    documented: set[str] = attributeNames_get(docstring)
    reasons: list[str] = [
        f"field {name} is not under Attributes:"
        for name in fieldNames_get(node)
        if name not in documented
    ]
    isPublic: bool = scope.isPublic and not node.name.startswith("_")
    if isPublic and not docstringSections_get(docstring) & EXAMPLE_SECTIONS:
        reasons.append("public class has no Example: section")
    return reasons


def moduleDocstring_check(tree: ast.Module, scope: Scope) -> list[str]:
    """Judge the module docstring against the documentation rules.

    Args:
        tree: A parsed module.
        scope: The module's scope; test modules need no sections.

    Returns:
        One reason per missing element; empty when the docstring passes.

    Example:
        >>> moduleDocstring_check(ast.parse("'''Tools.'''"), Scope(True, False))[0]
        'module docstring has no Key components: section'
    """
    docstring: str | None = ast.get_docstring(tree)
    if docstring is None:
        return ["missing module docstring"]
    if scope.isTestModule:
        return []
    sections: set[str] = docstringSections_get(docstring)
    return [
        f"module docstring has no {section}: section"
        for section in MODULE_SECTIONS
        if section not in sections
    ]


def definition_check(node: FunctionNode | ast.ClassDef, scope: Scope) -> list[str]:
    r"""Judge one definition's name and docstring.

    Args:
        node: A function, async function or class definition.
        scope: What encloses the definition.

    Returns:
        One reason per finding; empty when the definition passes.

    Example:
        >>> node = ast.parse("def x_of():\n    '''X.\n\n    Example:\n        x'''").body[0]
        >>> definition_check(node, Scope(isPublic=True, isTestModule=False))[0][:30]
        'the part after the underscore '
    """
    if isinstance(node, ast.ClassDef):
        return classDocstring_check(node, scope)
    nameReason: str | None = functionName_check(node)
    reasons: list[str] = [nameReason] if nameReason is not None else []
    return reasons + functionDocstring_check(node, scope)


def definitions_walk(
    nodes: list[ast.stmt], scope: Scope
) -> Iterator[tuple[FunctionNode | ast.ClassDef, Scope]]:
    r"""Visit every function and class definition, with the scope it sits in.

    Args:
        nodes: Statements to search, such as a module or class body.
        scope: The scope those statements sit in.

    Yields:
        Each definition with its scope, outermost first. A private class makes
        everything in it private; anything inside a function is private.

    Example:
        >>> tree = ast.parse("class _A:\n    def b_get(self): pass")
        >>> walked = definitions_walk(tree.body, Scope(True, False))
        >>> [(node.name, inner.isPublic) for node, inner in walked]
        [('_A', True), ('b_get', False)]
    """
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node, scope
            isClass: bool = isinstance(node, ast.ClassDef)
            innerPublic: bool = scope.isPublic and isClass and not node.name.startswith("_")
            yield from definitions_walk(node.body, Scope(innerPublic, scope.isTestModule))


def violations_collect(path: Path) -> Iterator[Violation]:
    """Walk one source file and judge everything in it.

    Args:
        path: A Python source file.

    Yields:
        Each finding, module docstring first, then definitions in source order.

    Raises:
        SyntaxError: If the file does not parse.
        OSError: If the file cannot be read.

    Example:
        >>> list(violations_collect(Path("style_check.py")))  # doctest: +SKIP
        []
    """
    tree: ast.Module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scope: Scope = Scope(isPublic=True, isTestModule=bool(TEST_MODULE_PATTERN.match(path.name)))
    for reason in moduleDocstring_check(tree, scope):
        yield Violation(path, 1, "<module>", reason)
    for node, nodeScope in definitions_walk(tree.body, scope):
        for reason in definition_check(node, nodeScope):
            yield Violation(path, node.lineno, node.name, reason)


def sources_find(roots: list[Path]) -> Iterator[Path]:
    """Expand files and directories into the Python files under them.

    Args:
        roots: Files and directories named on the command line.

    Yields:
        Each ``.py`` file, files as given and directories searched recursively
        in sorted order.

    Example:
        >>> sorted(sources_find([Path("example")]))  # doctest: +SKIP
        [PosixPath('example/transcript.py')]
    """
    for root in roots:
        if root.is_dir():
            yield from sorted(root.rglob("*.py"))
        else:
            yield root


def main() -> int:
    """Check every Python file named on the command line.

    Returns:
        ``EXIT_CLEAN`` when every check passes, ``EXIT_VIOLATIONS`` otherwise.
        Findings are printed one per line to standard error.

    Example:
        >>> sys.argv = ["style_check.py", "example/"]  # doctest: +SKIP
        >>> main()  # doctest: +SKIP
        0
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("paths", nargs="+", type=Path, help="files or directories to check")
    arguments: argparse.Namespace = parser.parse_args()
    violations: list[Violation] = [
        violation
        for path in sources_find(arguments.paths)
        for violation in violations_collect(path)
    ]
    for violation in violations:
        print(violation.report, file=sys.stderr)
    return EXIT_VIOLATIONS if violations else EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
