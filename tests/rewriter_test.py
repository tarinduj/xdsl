from io import StringIO
from typing import Callable

from xdsl.dialects.arith import Arith, Constant, Addi
from xdsl.dialects.builtin import ModuleOp, Builtin, i32
from xdsl.dialects.scf import Scf
from xdsl.ir import MLContext
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.rewriter import Rewriter


def rewrite_and_compare(prog: str, expected_prog: str,
                        transformation: Callable[[ModuleOp, Rewriter], None]):
    ctx = MLContext()
    builtin = Builtin(ctx)
    arith = Arith(ctx)
    scf = Scf(ctx)

    parser = Parser(ctx, prog)
    module = parser.parse_op()

    rewriter = Rewriter()
    transformation(module, rewriter)
    file = StringIO("")
    printer = Printer(stream=file)
    printer.print_op(module)
    assert file.getvalue().strip() == expected_prog.strip()


def test_operation_deletion():
    """Test rewrites where SSA values are deleted."""

    prog = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 5 : !i32]
}"""

    expected = \
"""module() {}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        rewriter.erase_op(constant_op)

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement
def test_replace_op_one_op():
    prog = \
    """module() {
%0 : !i32 = arith.constant() ["value" = 42 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 43 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        new_constant_op = Constant.from_int_constant(43, i32)
        rewriter.replace_op(constant_op, new_constant_op)

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement with multiple ops
def test_replace_op_multiple_op():
    prog = \
    """module() {
%0 : !i32 = arith.constant() ["value" = 2 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 1 : !i32]
  %1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
  %2 : !i32 = arith.addi(%1 : !i32, %1 : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        constant_op = module.ops[0]
        new_constant = Constant.from_int_constant(1, i32)
        new_add = Addi.get(new_constant, new_constant)

        rewriter.replace_op(constant_op, [new_constant, new_add])

    rewrite_and_compare(prog, expected, transformation)


# Test an operation replacement with manually specified results
def test_replace_op_new_results():
    prog = \
    """module() {
%0 : !i32 = arith.constant() ["value" = 2 : !i32]
%1 : !i32 = arith.addi(%0 : !i32, %0 : !i32)
%2 : !i32 = arith.muli(%1 : !i32, %1 : !i32)
}"""

    expected = \
"""module() {
  %0 : !i32 = arith.constant() ["value" = 2 : !i32]
  %1 : !i32 = arith.muli(%0 : !i32, %0 : !i32)
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        add_op = module.ops[1]

        rewriter.replace_op(add_op, [], [add_op.input1])

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_at_pos():
    """Test the inlining of a block at a certain position."""
    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {}
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        if_op = module.ops[1]
        module_block = module.regions[0].blocks[0]
        if_block = if_op.regions[0].blocks[0]

        rewriter.inline_block_at_pos(if_block, module_block, 1)

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_before():
    """Test the inlining of a block before an operation."""
    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {}
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        if_op = module.ops[1]
        if_block = if_op.regions[0].blocks[0]

        rewriter.inline_block_before(if_block, if_op)

    rewrite_and_compare(prog, expected, transformation)


def test_inline_block_after():
    """Test the inlining of a block after an operation."""
    prog = \
    """module() {
%0 : !i1 = arith.constant() ["value" = 1 : !i1]
scf.if(%0 : !i1) {
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
}
}"""

    expected = \
"""module() {
  %0 : !i1 = arith.constant() ["value" = 1 : !i1]
  %1 : !i32 = arith.constant() ["value" = 2 : !i32]
  scf.if(%0 : !i1) {}
}"""

    def transformation(module: ModuleOp, rewriter: Rewriter) -> None:
        if_op = module.ops[1]
        constant_op = module.ops[0]
        if_block = if_op.regions[0].blocks[0]

        rewriter.inline_block_after(if_block, constant_op)

    rewrite_and_compare(prog, expected, transformation)