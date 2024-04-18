from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence, Set
from io import StringIO
from typing import IO, ClassVar, Generic, TypeAlias, TypeVar

from typing_extensions import Self

from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    Signedness,
    StringAttr,
)
from xdsl.ir import (
    Attribute,
    Data,
    Dialect,
    Operation,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    IRDLOperation,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_attr_def,
    opt_operand_def,
    opt_result_def,
    result_def,
)
from xdsl.parser import AttrParser, Parser, UnresolvedOperand
from xdsl.printer import Printer
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

ASM_SYNTAX = "att" # "att" or "intel"

class X86RegisterType(Data[str], TypeAttribute, ABC):
    """
    An x86 register type.
    """

    _unallocated: ClassVar[Self | None] = None

    @classmethod
    def unallocated(cls) -> Self:
        if cls._unallocated is None:
            cls._unallocated = cls("")
        return cls._unallocated

    @property
    def register_name(self) -> str:
        """Returns name if allocated, raises ValueError if not"""
        if not self.is_allocated:
            raise ValueError("Cannot get name for unallocated register")
        return self.data

    @property
    def is_allocated(self) -> bool:
        """Returns true if an x86 register is allocated, otherwise false"""
        return bool(self.data)

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            name = parser.parse_optional_identifier()
            if name is None:
                return ""
            if not name.startswith("e") and not name.startswith("r"):
                assert name in cls.abi_index_by_name(), f"{name}"
            return name

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string(self.data)

    def verify(self) -> None:
        name = self.data
        if not self.is_allocated or name.startswith("e") or name.startswith("r"):
            return
        if name not in type(self).abi_index_by_name():
            raise VerifyException(f"{name} not in {self.instruction_set_name()}")

    @classmethod
    @abstractmethod
    def instruction_set_name(cls) -> str:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        raise NotImplementedError()


@irdl_attr_definition
class GeneralRegisterType(X86RegisterType):
    """
    An x86 register type.
    """

    name = "x86.reg"
    base: GeneralRegisterType | None = None
    
    def __init__(self, data, base: GeneralRegisterType | None = None):
        super().__init__(data)  # Initialize the superclass first
        # TODO: This is hacky. Fix it.
        object.__setattr__(self, 'base', base) 

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return GeneralRegisterType.X86_INDEX_BY_NAME

    X86_INDEX_BY_NAME = {
        "rax": 0, "eax": 0, "ax": 0, "al": 0, "ah": 0,
        "rcx": 1, "ecx": 1, "cx": 1, "cl": 1, "ch": 1,
        "rdx": 2, "edx": 2, "dx": 2, "dl": 2, "dh": 2,
        "rbx": 3, "ebx": 3, "bx": 3, "bl": 3, "bh": 3,
        "rsp": 4, "esp": 4, "sp": 4, "spl": 4,
        "rbp": 5, "ebp": 5, "bp": 5, "bpl": 5,
        "rsi": 6, "esi": 6, "si": 6, "sil": 6,
        "rdi": 7, "edi": 7, "di": 7, "dil": 7,
        "r8": 8, "r8d": 8, "r8w": 8, "r8b": 8,
        "r9": 9, "r9d": 9, "r9w": 9, "r9b": 9,
        "r10": 10, "r10d": 10, "r10w": 10, "r10b": 10,
        "r11": 11, "r11d": 11, "r11w": 11, "r11b": 11,
        "r12": 12, "r12d": 12, "r12w": 12, "r12b": 12,
        "r13": 13, "r13d": 13, "r13w": 13, "r13b": 13,
        "r14": 14, "r14d": 14, "r14w": 14, "r14b": 14,
        "r15": 15, "r15d": 15, "r15w": 15, "r15b": 15,
    }
    
@irdl_attr_definition
class MaskRegisterType(X86RegisterType):
    """
    An x86 register type for AVX512 instructions.
    """

    name = "x86.maskreg"

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86SIMD"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return MaskRegisterType.X86MASK_INDEX_BY_NAME

    X86MASK_INDEX_BY_NAME = {
        "k0": 0,
        "k1": 1,
        "k2": 2,
        "k3": 3,
        "k4": 4,
        "k5": 5,
        "k6": 6,
        "k7": 7,
    }


@irdl_attr_definition
class SIMDRegisterType(X86RegisterType):
    """
    An x86 register type for AVX512 instructions.
    """

    name = "x86.simdreg"
    

    @classmethod
    def instruction_set_name(cls) -> str:
        return "x86SIMD"

    @classmethod
    def abi_index_by_name(cls) -> dict[str, int]:
        return SIMDRegisterType.X86SIMD_INDEX_BY_NAME

    X86SIMD_INDEX_BY_NAME = {
        "zmm0": 0,
        "zmm1": 1,
        "zmm2": 2,
        "zmm3": 3,
        "zmm4": 4,
        "zmm5": 5,
        "zmm6": 6,
        "zmm7": 7,
        "zmm8": 8,
        "zmm9": 9,
        "zmm10": 10,
        "zmm11": 11,
        "zmm12": 12,
        "zmm13": 13,
        "zmm14": 14,
        "zmm15": 15,
        "zmm16": 16,
        "zmm17": 17,
        "zmm18": 18,
        "zmm19": 19,
        "zmm20": 20,
        "zmm21": 21,
        "zmm22": 22,
        "zmm23": 23,
        "zmm24": 24,
        "zmm25": 25,
        "zmm26": 26,
        "zmm27": 27,
        "zmm28": 28,
        "zmm29": 29,
        "zmm30": 30,
        "zmm31": 31,
    }


R1InvT = TypeVar("R1InvT", bound=X86RegisterType)
R2InvT = TypeVar("R2InvT", bound=X86RegisterType)
R3InvT = TypeVar("R3InvT", bound=X86RegisterType)
# RDInvT = TypeVar("RDInvT", bound=X86RegisterType)
# RSInvT = TypeVar("RSInvT", bound=X86RegisterType)
# RS1InvT = TypeVar("RS1InvT", bound=X86RegisterType)
# RS2InvT = TypeVar("RS2InvT", bound=X86RegisterType)


class Registers(ABC):
    """Namespace for named register constants."""

    # RAX Register and its subregisters
    RAX = GeneralRegisterType("rax")
    EAX = GeneralRegisterType("eax", base=RAX)
    AX = GeneralRegisterType("ax", base=EAX)
    AL = GeneralRegisterType("al", base=AX)
    AH = GeneralRegisterType("ah", base=AX)

    # RCX Register and its subregisters
    RCX = GeneralRegisterType("rcx")
    ECX = GeneralRegisterType("ecx", base=RCX)
    CX = GeneralRegisterType("cx", base=ECX)
    CL = GeneralRegisterType("cl", base=CX)
    CH = GeneralRegisterType("ch", base=CX)

    # RDX Register and its subregisters
    RDX = GeneralRegisterType("rdx")
    EDX = GeneralRegisterType("edx", base=RDX)
    DX = GeneralRegisterType("dx", base=EDX)
    DL = GeneralRegisterType("dl", base=DX)
    DH = GeneralRegisterType("dh", base=DX)

    # RBX Register and its subregisters
    RBX = GeneralRegisterType("rbx")
    EBX = GeneralRegisterType("ebx", base=RBX)
    BX = GeneralRegisterType("bx", base=EBX)
    BL = GeneralRegisterType("bl", base=BX)
    BH = GeneralRegisterType("bh", base=BX)

    # RSP Register and its subregisters
    RSP = GeneralRegisterType("rsp")
    ESP = GeneralRegisterType("esp", base=RSP)
    SP = GeneralRegisterType("sp", base=ESP)
    SPL = GeneralRegisterType("spl", base=SP)

    # RBP Register and its subregisters
    RBP = GeneralRegisterType("rbp")
    EBP = GeneralRegisterType("ebp", base=RBP)
    BP = GeneralRegisterType("bp", base=EBP)
    BPL = GeneralRegisterType("bpl", base=BP)

    # RSI Register and its subregisters
    RSI = GeneralRegisterType("rsi")
    ESI = GeneralRegisterType("esi", base=RSI)
    SI = GeneralRegisterType("si", base=ESI)
    SIL = GeneralRegisterType("sil", base=SI)

    # RDI Register and its subregisters
    RDI = GeneralRegisterType("rdi")
    EDI = GeneralRegisterType("edi", base=RDI)
    DI = GeneralRegisterType("di", base=EDI)
    DIL = GeneralRegisterType("dil", base=DI)

    # R8 to R15 Registers and their subregisters
    R8 = GeneralRegisterType("r8")
    R8D = GeneralRegisterType("r8d", base=R8)
    R8W = GeneralRegisterType("r8w", base=R8D)
    R8B = GeneralRegisterType("r8b", base=R8W)

    R9 = GeneralRegisterType("r9")
    R9D = GeneralRegisterType("r9d", base=R9)
    R9W = GeneralRegisterType("r9w", base=R9D)
    R9B = GeneralRegisterType("r9b", base=R9W)

    R10 = GeneralRegisterType("r10")
    R10D = GeneralRegisterType("r10d", base=R10)
    R10W = GeneralRegisterType("r10w", base=R10D)
    R10B = GeneralRegisterType("r10b", base=R10W)

    R11 = GeneralRegisterType("r11")
    R11D = GeneralRegisterType("r11d", base=R11)
    R11W = GeneralRegisterType("r11w", base=R11D)
    R11B = GeneralRegisterType("r11b", base=R11W)

    R12 = GeneralRegisterType("r12")
    R12D = GeneralRegisterType("r12d", base=R12)
    R12W = GeneralRegisterType("r12w", base=R12D)
    R12B = GeneralRegisterType("r12b", base=R12W)

    R13 = GeneralRegisterType("r13")
    R13D = GeneralRegisterType("r13d", base=R13)
    R13W = GeneralRegisterType("r13w", base=R13D)
    R13B = GeneralRegisterType("r13b", base=R13W)
    
    R14 = GeneralRegisterType("r14")
    R14D = GeneralRegisterType("r14d", base=R14)
    R14W = GeneralRegisterType("r14w", base=R14D)
    R14B = GeneralRegisterType("r14b", base=R14W)
    
    R15 = GeneralRegisterType("r15")
    R15D = GeneralRegisterType("r15d", base=R15)
    R15W = GeneralRegisterType("r15w", base=R15D)
    R15B = GeneralRegisterType("r15b", base=R15W)

    ZMM0 = SIMDRegisterType("zmm0")
    ZMM1 = SIMDRegisterType("zmm1")
    ZMM2 = SIMDRegisterType("zmm2")
    ZMM3 = SIMDRegisterType("zmm3")
    ZMM4 = SIMDRegisterType("zmm4")
    ZMM5 = SIMDRegisterType("zmm5")
    ZMM6 = SIMDRegisterType("zmm6")
    ZMM7 = SIMDRegisterType("zmm7")
    ZMM8 = SIMDRegisterType("zmm8")
    ZMM9 = SIMDRegisterType("zmm9")
    ZMM10 = SIMDRegisterType("zmm10")
    ZMM11 = SIMDRegisterType("zmm11")
    ZMM12 = SIMDRegisterType("zmm12")
    ZMM13 = SIMDRegisterType("zmm13")
    ZMM14 = SIMDRegisterType("zmm14")
    ZMM15 = SIMDRegisterType("zmm15")
    ZMM16 = SIMDRegisterType("zmm16")
    ZMM17 = SIMDRegisterType("zmm17")
    ZMM18 = SIMDRegisterType("zmm18")
    ZMM19 = SIMDRegisterType("zmm19")
    ZMM20 = SIMDRegisterType("zmm20")
    ZMM21 = SIMDRegisterType("zmm21")
    ZMM22 = SIMDRegisterType("zmm22")
    ZMM23 = SIMDRegisterType("zmm23")
    ZMM24 = SIMDRegisterType("zmm24")
    ZMM25 = SIMDRegisterType("zmm25")
    ZMM26 = SIMDRegisterType("zmm26")
    ZMM27 = SIMDRegisterType("zmm27")
    ZMM28 = SIMDRegisterType("zmm28")
    ZMM29 = SIMDRegisterType("zmm29")
    ZMM30 = SIMDRegisterType("zmm30")
    ZMM31 = SIMDRegisterType("zmm31")
    
    K0 = MaskRegisterType("k0")
    K1 = MaskRegisterType("k1")
    K2 = MaskRegisterType("k2")
    K3 = MaskRegisterType("k3")
    K4 = MaskRegisterType("k4")
    K5 = MaskRegisterType("k5")
    K6 = MaskRegisterType("k6")
    K7 = MaskRegisterType("k7")


@irdl_attr_definition
class LabelAttr(Data[str]):
    name = "x86.label"

    @classmethod
    def parse_parameter(cls, parser: AttrParser) -> str:
        with parser.in_angle_brackets():
            return parser.parse_str_literal()

    def print_parameter(self, printer: Printer) -> None:
        with printer.in_angle_brackets():
            printer.print_string_literal(self.data)


class X86Op(Operation, ABC):
    """
    Base class for operations that can be a part of x86 assembly printing.
    """

    @abstractmethod
    def assembly_line(self) -> str | None:
        raise NotImplementedError()

    @classmethod
    def parse(cls, parser: Parser) -> Self:
        args = cls.parse_unresolved_operands(parser)
        custom_attributes = cls.custom_parse_attributes(parser)
        remaining_attributes = parser.parse_optional_attr_dict()
        # TODO ensure distinct keys for attributes
        attributes = custom_attributes | remaining_attributes
        regions = parser.parse_region_list()
        pos = parser.pos
        operand_types, result_types = cls.parse_op_type(parser)
        operands = parser.resolve_operands(args, operand_types, pos)
        return cls.create(
            operands=operands,
            result_types=result_types,
            attributes=attributes,
            regions=regions,
        )

    @classmethod
    def parse_unresolved_operands(cls, parser: Parser) -> list[UnresolvedOperand]:
        """
        Parse a list of comma separated unresolved operands.

        Notice that this method will consume trailing comma.
        """
        if operand := parser.parse_optional_unresolved_operand():
            operands = [operand]
            while parser.parse_optional_punctuation(",") and (
                operand := parser.parse_optional_unresolved_operand()
            ):
                operands.append(operand)
            return operands
        return []

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        """
        Parse attributes with custom syntax. Subclasses may override this method.
        """
        return parser.parse_optional_attr_dict()

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        parser.parse_punctuation(":")
        func_type = parser.parse_function_type()
        return func_type.inputs.data, func_type.outputs.data

    def print(self, printer: Printer) -> None:
        if self.operands:
            printer.print(" ")
            printer.print_list(self.operands, printer.print_operand)
        printed_attributes = self.custom_print_attributes(printer)
        unprinted_attributes = {
            name: attr
            for name, attr in self.attributes.items()
            if name not in printed_attributes
        }
        printer.print_op_attributes(unprinted_attributes)
        printer.print_regions(self.regions)
        self.print_op_type(printer)

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        """
        Print attributes with custom syntax. Return the names of the attributes printed. Subclasses may override this method.
        """
        printer.print_op_attributes(self.attributes)
        return self.attributes.keys()

    def print_op_type(self, printer: Printer) -> None:
        printer.print(" : ")
        printer.print_operation_type(self)


AssemblyInstructionArg: TypeAlias = (
    AnyIntegerAttr | LabelAttr | SSAValue | GeneralRegisterType | str | int
)


class X86Instruction(X86Op):
    """
    Base class for operations that can be a part of x86 assembly printing. Must
    represent an instruction in the x86 instruction set.

    The name of the operation will be used as the x86 assembly instruction name.
    """

    comment: StringAttr | None = opt_attr_def(StringAttr)
    """
    An optional comment that will be printed along with the instruction.
    """

    @abstractmethod
    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        """
        The arguments to the instruction, in the order they should be printed in the
        assembly.
        """
        raise NotImplementedError()

    def assembly_instruction_name(self) -> str:
        """
        By default, the name of the instruction is the same as the name of the operation.
        """

        return self.name.split(".", 1)[-1]

    def assembly_line(self) -> str | None:
        # default assembly code generator
        instruction_name = self.assembly_instruction_name()
        if ASM_SYNTAX == "att":
            arg_str = ", ".join(
                _assembly_arg_str(arg)
                for arg in self.assembly_line_args()[::-1]
                if arg is not None
            )
        else:
            arg_str = ", ".join(
                _assembly_arg_str(arg)
                for arg in self.assembly_line_args()
                if arg is not None
            )
        return _assembly_line(instruction_name, arg_str, self.comment)


class SingleOperandInstruction(IRDLOperation, X86Instruction, ABC):
    """
    Base class for instructions that take a single operand.
    """


class DoubleOperandInstruction(IRDLOperation, X86Instruction, ABC):
    """
    Base class for instructions that take two operands.
    """


class TripleOperandInstruction(IRDLOperation, X86Instruction, ABC):
    """
    Base class for instructions that take three operands.
    """


class RROperation(Generic[R1InvT, R2InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have two registers.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2],
            attributes={
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2


class ROperation(Generic[R1InvT], SingleOperandInstruction):
    """
    A base class for x86 operations that have one register.
    """

    source = opt_operand_def(R1InvT)
    destination = opt_result_def(R1InvT)

    def __init__(
        self,
        source: Operation | SSAValue | None = None,
        *,
        comment: str | StringAttr | None = None,
        destination: R1InvT | None = None,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[source],
            attributes={
                "comment": comment,
            },
            result_types=[destination],
        )

    def verify_(self) -> None:
        if self.source is None and self.destination is None:
            raise VerifyException("Either source or destination must be specified")
        if self.source is not None and self.destination is not None:
            raise VerifyException("Cannot specify both source and destination")
        return super().verify_()

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return (self.destination,) if self.destination else (self.source,)


@irdl_op_definition
class AddOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Adds the registers r1 and r2 and stores the result in r1.

    x[r1] = x[r1] + x[r2]

    https://www.felixcloutier.com/x86/add
    """

    name = "x86.add"


@irdl_op_definition
class SubOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    subtracts r2 from r1 and stores the result in r1.

    x[r1] = x[r1] - x[r2]

    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.sub"


@irdl_op_definition
class ImulOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Multiplies the registers r1 and r2 and stores the result in r1.

    x[r1] = x[r1] * x[r2]

    https://www.felixcloutier.com/x86/imul
    """

    name = "x86.imul"


@irdl_op_definition
class IdivOp(ROperation[GeneralRegisterType]):
    """
    Divide rdx:rax by x[r1]. Store quotient in rax and store remainder in rdx.

    https://www.felixcloutier.com/x86/idiv
    """

    name = "x86.idiv"


@irdl_op_definition
class NotOp(ROperation[GeneralRegisterType]):
    """
    bitwise not of r1, stored in r1

    x[r1] = ~x[r1]

    https://www.felixcloutier.com/x86/not
    """

    name = "x86.not"


@irdl_op_definition
class AndOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise and of r1 and r2, stored in r1

    x[r1] = x[r1] & x[r2]

    https://www.felixcloutier.com/x86/and
    """

    name = "x86.and"


@irdl_op_definition
class OrOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise or of r1 and r2, stored in r1

    x[r1] = x[r1] | x[r2]

    https://www.felixcloutier.com/x86/or
    """

    name = "x86.or"


@irdl_op_definition
class XorOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    bitwise xor of r1 and r2, stored in r1

    x[r1] = x[r1] ^ x[r2]

    https://www.felixcloutier.com/x86/xor
    """

    name = "x86.xor"


@irdl_op_definition
class MovOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value of r1 into r2.

    x[r1] = x[r2]

    https://www.felixcloutier..com/x86/mov
    """

    name = "x86.mov"
    
@irdl_op_definition
class RRKmovbOp(RROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value of r1 into r2.

    x[r1] = x[r2]

    https://www.felixcloutier..com/x86/mov
    """

    name = "x86.kmovb"


@irdl_op_definition
class PushOp(ROperation[GeneralRegisterType]):
    """
    Decreases %rsp and places r1 at the new memory location pointed to by %rsp.

    https://www.felixcloutier.com/x86/push
    """

    name = "x86.push"

    def verify_(self) -> None:
        if self.source is None:
            raise VerifyException("Source register must be specified")
        else:
            return super().verify_()


@irdl_op_definition
class PopOp(ROperation[GeneralRegisterType]):
    """
    Copies the value at the top of the stack into r1 and increases %rsp.

    https://www.felixcloutier.com/x86/pop
    """

    name = "x86.pop"

    def verify_(self) -> None:
        if self.destination is None:
            raise VerifyException("Destination register must be specified")
        else:
            return super().verify_()


class RRROperation(Generic[R1InvT, R2InvT, R3InvT], TripleOperandInstruction):
    """
    A base class for x86 operations that have three registers.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    r3 = operand_def(R3InvT)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        r3: Operation | SSAValue,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2, r3],
            attributes={
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2, self.r3


class RMOperation(Generic[R1InvT, R2InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have a register and a memory offset.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None = None,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)  # I have no clue why that is 12
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["offset"] = _parse_immediate_value(
            parser, IntegerType(12, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
        return {"offset"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        if self.offset is not None and self.offset.value.data != 0:
            offset = self.offset.value.data
            if ASM_SYNTAX == "att":
                arg_str = f"{offset: #x}({source}), {destination}"
            else:
                arg_str = f"{destination}, [{source} + {offset}]"
            return _assembly_line(
                instruction_name, arg_str, self.comment
            )
        else:
            if ASM_SYNTAX == "att":
                arg_str = f"({source}), {destination}"
            else:
                arg_str = f"{destination}, [{source}]"
            return _assembly_line(
                instruction_name, arg_str, self.comment
            )


@irdl_op_definition
class RMMovOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from the memory location pointed to by r2 into r1.

    x[r1] = [x[r2]]

    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.mov"

class MOperation(Generic[R1InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have a register and a memory offset.
    """

    r1 = operand_def(R1InvT)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None = None,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)  # I have no clue why that is 12
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["offset"] = _parse_immediate_value(
            parser, IntegerType(12, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
        return {"offset"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        source = _assembly_arg_str(self.r1)
        if self.offset is not None and self.offset.value.data != 0:
            offset = self.offset.value.data
            if ASM_SYNTAX == "att":
                arg_str = f"{offset: #x}({source})"
            else:
                arg_str = f"[{source} + {offset}]"
            return _assembly_line(
                instruction_name, arg_str, self.comment
            )
        else:
            if ASM_SYNTAX == "att":
                arg_str = f"({source})"
            else:
                arg_str = f"[{source}]"
            return _assembly_line(
                instruction_name, arg_str, self.comment
            )
            
@irdl_op_definition
class MPrefetcht1Op(MOperation[GeneralRegisterType]):
    """
    Prefetches the memory location pointed to by r1 into the L1 cache.

    https://www.felixcloutier.com/x86/prefetch
    """

    name = "x86.prefetcht1"
    
@irdl_op_definition
class MJlOp(MOperation[GeneralRegisterType]):
    """
    Jump to the memory location pointed to by r1.

    https://www.felixcloutier.com/x86/jmp
    """

    name = "x86.jl"
    

    
@irdl_op_definition
class RMVbroadcastsdOp(RMOperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from the memory location pointed to by r2 into r1.

    x[r1] = [x[r2]]

    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.vbroadcastsd"

class RIOperation(Generic[R1InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have one register and an immediate value.
    """

    r1 = operand_def(R1InvT)
    immediate: AnyIntegerAttr | LabelAttr = attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        immediate: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, 32)  # 32 bits?
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1],
            attributes={
                "immediate": immediate,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.immediate
    
    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = _parse_immediate_value(
            parser, IntegerType(32, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        _print_immediate_value(printer, self.immediate)
        return {"immediate"}
    
@irdl_op_definition
class RIAddOp(RIOperation[GeneralRegisterType]):
    """
    Adds the register r1 and an immediate and stores the result in r1.

    x[r1] = x[r1] + i

    https://www.felixcloutier.com/x86/add
    """

    name = "x86.add"
    
@irdl_op_definition
class RICmpOp(RIOperation[GeneralRegisterType]):
    """
    Compares the register r1 and an immediate.

    https://www.felixcloutier.com/x86/cmp
    """

    name = "x86.cmp"
    
@irdl_op_definition
class RISubOp(RIOperation[GeneralRegisterType]):
    """
    Subtracts an immediate from r1 and stores the result in r1.

    x[r1] = x[r1] - i

    https://www.felixcloutier.com/x86/sub
    """

    name = "x86.sub"
    
@irdl_op_definition
class RIMovabsOp(RIOperation[GeneralRegisterType]):
    """
    Moves the immediate value into r1.

    x[r1] = i

    https://docs.oracle.com/cd/E18752_01/html/817-5477/ennbz.html#indexterm-150
    """

    name = "x86.movabs"
    
@irdl_op_definition
class RIMovOp(RIOperation[GeneralRegisterType]):
    """
    Copies the immediate value into r1.

    x[r1] = i

    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.mov"


class MIOperation(Generic[R1InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have one memory reference and an immediate value.
    """

    r1 = operand_def(R1InvT)
    immediate: AnyIntegerAttr | LabelAttr = attr_def(AnyIntegerAttr)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        immediate: int | AnyIntegerAttr,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(immediate, int):
            immediate = IntegerAttr(immediate, 32)  # 32 bits?
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1],
            attributes={
                "immediate": immediate,
                "offset": offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.immediate

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["immediate"] = _parse_immediate_value(
            parser, IntegerType(32, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        _print_immediate_value(printer, self.immediate)
        return {"immediate"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        immediate = _assembly_arg_str(self.immediate)
        if self.offset is not None:
            offset = _assembly_arg_str(self.offset)
            if ASM_SYNTAX == "att":
                arg_str = f"{offset: #x}({destination}), {immediate}"
            else:
                arg_str = f"[{destination} + {offset}], {immediate}"
            
            return _assembly_line(
                instruction_name,
                arg_str,
                self.comment,
            )
        else:
            return _assembly_line(
                instruction_name, f"[{destination}], {immediate}", self.comment
            )


@irdl_op_definition
class MIMovOp(MIOperation[GeneralRegisterType]):
    """
    Copies the immediate value into r1.

    [x[r1]] = immediate

    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.mov"


class MROperation(Generic[R1InvT, R2InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have one memory reference and one register.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2],
            attributes={
                "offset": offset,
                "comment": comment,
            },
            result_types=[result],
        )

    def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
        return self.r1, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["offset"] = _parse_immediate_value(
            parser, IntegerType(12, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
        return {"offset"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        if self.offset is not None and self.offset.value.data != 0:
            offset = self.offset.value.data
            if ASM_SYNTAX == "att":
                arg_str = f"{source}, {offset: #x}({destination})"
            else:
                arg_str = f"[{destination}  + {offset}], {source}"
            return _assembly_line(
                instruction_name, arg_str, self.comment
            )
        else:
            if ASM_SYNTAX == "att":
                arg_str = f"{source}, ({destination})"
            else:
                arg_str = f"[{destination}], {source}"
            return _assembly_line(
                instruction_name, arg_str, self.comment
            )


@irdl_op_definition
class MRMovOp(MROperation[GeneralRegisterType, GeneralRegisterType]):
    """
    Copies the value from r2 into the memory location pointed to by r1.

    x[r1] = [x[r2]]

    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.mov"
    
@irdl_op_definition
class MRVmovupdOp(MROperation[GeneralRegisterType, SIMDRegisterType]):
    """
    Copies the value from r2 into the memory location pointed to by r1.

    x[r1] = [x[r2]]

    https://www.felixcloutier.com/x86/mov
    """

    name = "x86.vmovupd"


@irdl_op_definition
class DirectiveOp(IRDLOperation, X86Op):
    """
    The directive operation is used to represent a directive in the assembly code. (e.g. .globl; .type etc)
    """

    name = "x86.directive"

    directive: StringAttr = attr_def(StringAttr)
    value: StringAttr | None = opt_attr_def(StringAttr)

    def __init__(
        self,
        directive: str | StringAttr,
        value: str | StringAttr | None,
    ):
        if isinstance(directive, str):
            directive = StringAttr(directive)
        if isinstance(value, str):
            value = StringAttr(value)

        super().__init__(
            attributes={
                "directive": directive,
                "value": value,
            },
        )

    def assembly_line(self) -> str | None:
        if self.value is not None and self.value.data:
            arg_str = _assembly_arg_str(self.value.data)
        else:
            arg_str = ""

        return _assembly_line(self.directive.data, arg_str, is_indented=False)

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["directive"] = StringAttr(
            parser.parse_str_literal("Expected directive")
        )
        if (value := parser.parse_optional_str_literal()) is not None:
            attributes["value"] = StringAttr(value)
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(" ")
        printer.print_string_literal(self.directive.data)
        if self.value is not None:
            printer.print(" ")
            printer.print_string_literal(self.value.data)
        return {"directive", "value"}

    def print_op_type(self, printer: Printer) -> None:
        return

    @classmethod
    def parse_op_type(
        cls, parser: Parser
    ) -> tuple[Sequence[Attribute], Sequence[Attribute]]:
        return (), ()


@irdl_op_definition
class Vfmadd231pdOp(RRROperation[SIMDRegisterType, SIMDRegisterType, SIMDRegisterType]):
    """
    Multiply packed double-precision floating-point elements in r2 and r3, add the intermediate result to r1, and store the final result in r1.

    https://www.felixcloutier.com/x86/vfmadd132pd:vfmadd213pd:vfmadd231pd
    """

    name = "x86.vfmadd231pd"


@irdl_op_definition
class RMVmovapdOp(RMOperation[SIMDRegisterType, GeneralRegisterType]):
    """
    Move aligned packed double-precision floating-point elements.

    https://www.felixcloutier.com/x86/movapd
    """

    name = "x86.vmovapd"
    
@irdl_op_definition
class RMVmovupdOp(RMOperation[SIMDRegisterType, GeneralRegisterType]):
    """
    Move aligned packed double-precision floating-point elements.

    https://www.felixcloutier.com/x86/movapd
    """

    name = "x86.vmovupd"


@irdl_op_definition
class VbroadcastsdOp(RMOperation[SIMDRegisterType, GeneralRegisterType]):
    """
    Broadcast scalar double-precision floating-point element.

    https://www.felixcloutier.com/x86/vbroadcast
    """

    name = "x86.vbroadcastsd"


# region Assembly printing

class RKMOperation(Generic[R1InvT, R2InvT, R3InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have a register with a mask and a memory reference.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    r3 = operand_def(R3InvT)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)
    modifier: StringAttr | None = opt_attr_def(StringAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        r3: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        modifier: str | StringAttr | None,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)
        if isinstance(modifier, str):
            modifier = StringAttr(modifier)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2, r3],
            attributes={
                "offset": offset,
                "modifier": modifier,
                "comment": comment,
            },
            result_types=[result],
        )

    # def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
    #     return self.r1, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["offset"] = _parse_immediate_value(
            parser, IntegerType(12, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
        return {"offset"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        mask = _assembly_arg_str(self.r3)
        if self.offset is not None and self.offset.value.data != 0:
            offset = self.offset.value.data
            if self.modifier is not None:
                modifier = self.modifier.data   
                if ASM_SYNTAX == "att":
                    arg_str = f"{offset: #x}({source}), {destination}{{{mask}}}{{{modifier}}}"
                else:
                    arg_str = f"{destination}, [{source} + {offset}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
            else: 
                if ASM_SYNTAX == "att":
                    arg_str = f"{offset: #x}({source}),  {destination}{{{mask}}}"
                else:
                    arg_str = f"{destination}, [{source} + {offset}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
        else:
            if self.modifier is not None:
                modifier = self.modifier.data
                if ASM_SYNTAX == "att":
                    arg_str = f"({source}),  {destination}{{{mask}}}{{{modifier}}}"
                else:
                    arg_str = f"{destination}, [{source}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
            else:
                if ASM_SYNTAX == "att":
                    arg_str = f"({source}),  {destination}{{{mask}}}"
                else:
                    arg_str = f"{destination}, [{source}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
                
class RKMOperation(Generic[R1InvT, R2InvT, R3InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have a register with a mask and a memory reference.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    r3 = operand_def(R3InvT)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)
    modifier: StringAttr | None = opt_attr_def(StringAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        r3: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        modifier: str | StringAttr | None,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)
        if isinstance(modifier, str):
            modifier = StringAttr(modifier)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2, r3],
            attributes={
                "offset": offset,
                "modifier": modifier,
                "comment": comment,
            },
            result_types=[result],
        )

    # def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
    #     return self.r1, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["offset"] = _parse_immediate_value(
            parser, IntegerType(12, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
        return {"offset"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        mask = _assembly_arg_str(self.r3)
        if self.offset is not None and self.offset.value.data != 0:
            offset = self.offset.value.data
            if self.modifier is not None:
                modifier = self.modifier.data   
                if ASM_SYNTAX == "att":
                    arg_str = f"{offset: #x}({source}), {destination}{{{mask}}}{{{modifier}}}"
                else:
                    arg_str = f"{destination}, [{source} + {offset}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
            else: 
                if ASM_SYNTAX == "att":
                    arg_str = f"{offset: #x}({source}),  {destination}{{{mask}}}"
                else:
                    arg_str = f"{destination}, [{source} + {offset}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
        else:
            if self.modifier is not None:
                modifier = self.modifier.data
                if ASM_SYNTAX == "att":
                    arg_str = f"({source}),  {destination}{{{mask}}}{{{modifier}}}"
                else:
                    arg_str = f"{destination}, [{source}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
            else:
                if ASM_SYNTAX == "att":
                    arg_str = f"({source}),  {destination}{{{mask}}}"
                else:
                    arg_str = f"{destination}, [{source}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )

@irdl_op_definition
class RKMVmovupdOp(RKMOperation[SIMDRegisterType, GeneralRegisterType, MaskRegisterType]):
    """
    Move aligned packed double-precision floating-point elements.

    https://www.felixcloutier.com/x86/movapd
    """

    name = "x86.vmovupd"
    
class MKROperation(Generic[R1InvT, R2InvT, R3InvT], DoubleOperandInstruction):
    """
    A base class for x86 operations that have a register with a mask and a memory reference.
    """

    r1 = operand_def(R1InvT)
    r2 = operand_def(R2InvT)
    r3 = operand_def(R3InvT)
    offset: AnyIntegerAttr | None = opt_attr_def(AnyIntegerAttr)
    modifier: StringAttr | None = opt_attr_def(StringAttr)

    result = result_def(R1InvT)

    def __init__(
        self,
        r1: Operation | SSAValue,
        r2: Operation | SSAValue,
        r3: Operation | SSAValue,
        offset: int | AnyIntegerAttr | None,
        modifier: str | StringAttr | None,
        *,
        comment: str | StringAttr | None = None,
        result: R1InvT,
    ):
        if isinstance(offset, int):
            offset = IntegerAttr(offset, 12)
        if isinstance(modifier, str):
            modifier = StringAttr(modifier)
        if isinstance(comment, str):
            comment = StringAttr(comment)

        super().__init__(
            operands=[r1, r2, r3],
            attributes={
                "offset": offset,
                "modifier": modifier,
                "comment": comment,
            },
            result_types=[result],
        )

    # def assembly_line_args(self) -> tuple[AssemblyInstructionArg | None, ...]:
    #     return self.r1, self.r2

    @classmethod
    def custom_parse_attributes(cls, parser: Parser) -> dict[str, Attribute]:
        attributes = dict[str, Attribute]()
        attributes["offset"] = _parse_immediate_value(
            parser, IntegerType(12, Signedness.SIGNED)
        )
        return attributes

    def custom_print_attributes(self, printer: Printer) -> Set[str]:
        printer.print(", ")
        if self.offset is not None:
            _print_immediate_value(printer, self.offset)
        return {"offset"}

    def assembly_line(self) -> str | None:
        instruction_name = self.assembly_instruction_name()
        destination = _assembly_arg_str(self.r1)
        source = _assembly_arg_str(self.r2)
        mask = _assembly_arg_str(self.r3)
        if self.offset is not None and self.offset.value.data != 0:
            offset = self.offset.value.data
            if self.modifier is not None:
                modifier = self.modifier.data   
                if ASM_SYNTAX == "att":
                    arg_str = f"{source}, {offset: #x}({destination}){{{mask}}}{{{modifier}}}"
                else:
                    arg_str = f"{destination}, [{source} + {offset}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
            else: 
                if ASM_SYNTAX == "att":
                    arg_str = f"{source}, {offset: #x}({destination}){{{mask}}}"
                else:
                    arg_str = f"{destination}, [{source} + {offset}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
        else:
            if self.modifier is not None:
                modifier = self.modifier.data
                if ASM_SYNTAX == "att":
                    arg_str = f"{source}, ({destination}){{{mask}}}{{{modifier}}}"
                else:
                    arg_str = f"{destination}, [{source}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )
            else:
                if ASM_SYNTAX == "att":
                    arg_str = f"{source}, ({destination}){{{mask}}}"
                else:
                    arg_str = f"{destination}, [{source}]"
                return _assembly_line(
                    instruction_name, arg_str, self.comment
                )

@irdl_op_definition
class MKRVmovupdOp(MKROperation[SIMDRegisterType, GeneralRegisterType, MaskRegisterType]):
    """
    Move aligned packed double-precision floating-point elements.

    https://www.felixcloutier.com/x86/movapd
    """

    name = "x86.vmovupd"

def _append_comment(line: str, comment: StringAttr | None) -> str:
    if comment is None:
        return line

    padding = " " * max(0, 48 - len(line))

    return f"{line}{padding} # {comment.data}"


def _assembly_arg_str(arg: AssemblyInstructionArg) -> str:
    if isa(arg, AnyIntegerAttr):
        if ASM_SYNTAX == "att":
            return f"${arg.value.data:#x}"
        else:
            return f"{arg.value.data}"
    elif isinstance(arg, int):
        if ASM_SYNTAX == "att":
            return f"${arg:#x}"
        else:
            return f"{arg}"
    elif isinstance(arg, LabelAttr):
        return arg.data
    elif isinstance(arg, str):
        return arg
    elif isinstance(arg, GeneralRegisterType | SIMDRegisterType | MaskRegisterType):
        if ASM_SYNTAX == "att":
            return f"%{arg.register_name}"
        else:
            return arg.register_name
    elif isinstance(arg.type, GeneralRegisterType | SIMDRegisterType | MaskRegisterType):
        if ASM_SYNTAX == "att":
            return f"%{arg.type.register_name}"
        else:
            return arg.type.register_name
    else:
        assert False, f"{arg.type}"


def _assembly_line(
    name: str,
    arg_str: str,
    comment: StringAttr | None = None,
    is_indented: bool = True,
) -> str:
    code = "    " if is_indented else ""
    code += name
    if arg_str:
        code += f" {arg_str}"
    code = _append_comment(code, comment)
    return code


def print_assembly(module: ModuleOp, output: IO[str]) -> None:
    for op in module.body.walk():
        assert isinstance(op, X86Op), f"{op}"
        asm = op.assembly_line()
        if asm is not None:
            print(asm, file=output)


def x86_code(module: ModuleOp) -> str:
    stream = StringIO()
    print_assembly(module, stream)
    return stream.getvalue()


def _parse_optional_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr | None:
    """
    Parse an optional immediate value. If an integer is parsed, an integer attr with the specified type is created.
    """
    if (immediate := parser.parse_optional_integer()) is not None:
        return IntegerAttr(immediate, integer_type)
    if (immediate := parser.parse_optional_str_literal()) is not None:
        return LabelAttr(immediate)


def _parse_immediate_value(
    parser: Parser, integer_type: IntegerType | IndexType
) -> IntegerAttr[IntegerType | IndexType] | LabelAttr:
    return parser.expect(
        lambda: _parse_optional_immediate_value(parser, integer_type),
        "Expected immediate",
    )


def _print_immediate_value(printer: Printer, immediate: AnyIntegerAttr | LabelAttr):
    match immediate:
        case IntegerAttr():
            printer.print(immediate.value.data)
        case LabelAttr():
            printer.print_string_literal(immediate.data)


class GetAnyRegisterOperation(Generic[R1InvT], IRDLOperation, X86Op):
    """
    This instruction allows us to create an SSAValue with for a given register name. This
    is useful for bridging the x86 convention that stores the result of function calls
    in `eax` into SSA form.

    """

    result = result_def(R1InvT)

    def __init__(
        self,
        register_type: R1InvT,
    ):
        super().__init__(result_types=[register_type])

    def assembly_line(self) -> str | None:
        return None


@irdl_op_definition
class GetRegisterOp(GetAnyRegisterOperation[GeneralRegisterType]):
    name = "x86.get_register"


@irdl_op_definition
class GetAVXRegisterOp(GetAnyRegisterOperation[SIMDRegisterType]):
    name = "x86.get_avx_register"


X86 = Dialect(
    "x86",
    [
        AddOp,
        SubOp,
        ImulOp,
        IdivOp,
        NotOp,
        AndOp,
        OrOp,
        XorOp,
        MovOp,
        PushOp,
        PopOp,
        Vfmadd231pdOp,
        RMVmovapdOp,
        VbroadcastsdOp,
        DirectiveOp,
        GetRegisterOp,
        GetAVXRegisterOp,
    ],
    [
        GeneralRegisterType,
        MaskRegisterType,
        SIMDRegisterType,
        LabelAttr,
    ],
)
# endregion
