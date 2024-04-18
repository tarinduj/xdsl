from xdsl.dialects import x86
from xdsl.utils.exceptions import VerifyException
from xdsl.ir import Block, Operation, SSAValue
from xdsl.dialects.builtin import IntegerAttr, i64

PARSER_ASM_SYNTAX = "att"

class ParserSSAValue(SSAValue):
    @property
    def owner(self) -> Operation | Block:
        assert False, "Attempting to get the owner of a `ParserSSAValue`"

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

def get_RPushOp(r1):
    r1 = ParserSSAValue(r1)
    
    push_op = x86.PushOp(source=r1)
    return push_op

def get_RRAddOp(r1, r2):
    print(r1, r2)
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)
    
    add_op = x86.AddOp(r1, r2, result=r1)
    return add_op

def get_RIAddOp(r1, i):
    r1 = ParserSSAValue(r1)
    i = IntegerAttr(i, i64)
    
    add_op = x86.RIAddOp(r1, i, result=r1)
    return add_op

def get_RISubOp(r1, i):
    r1 = ParserSSAValue(r1)
    i = IntegerAttr(i, i64)
    
    sub_op = x86.RISubOp(r1, i, result=r1)
    return sub_op

def get_RRMovOp(r1, r2):
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)

    mov_op = x86.MovOp(r1, r2, result=r1)
    return mov_op

def get_RMMovOp(r1, arg2):
    r2, offset = arg2
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)
    offset = IntegerAttr(offset, i64)
    
    mov_op = x86.RMMovOp(r1, r2, offset, result=r1)
    return mov_op

def get_RIMovOp(r1, i):
    r1 = ParserSSAValue(r1)
    i = IntegerAttr(i, i64)
    
    mov_op = x86.RIMovOp(r1, i, result=r1)
    return mov_op

def get_MRMovOp(arg1, r2):
    r1, offset = arg1
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)
    offset = IntegerAttr(offset, i64)
    
    mov_op = x86.MRMovOp(r1, r2, offset, result=r1)
    return mov_op

def get_RIMovabsOp(r1, i):
    r1 = ParserSSAValue(r1)
    i = IntegerAttr(i, i64)
    
    movabs_op = x86.RIMovabsOp(r1, i, result=r1)
    return movabs_op

def get_RRKmovbOp(r1, r2):
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)

    kmovb_op = x86.RRKmovbOp(r1, r2, result=r1)
    return kmovb_op

def get_RRAndOp(r1, r2):
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)

    and_op = x86.AndOp(r1, r2, result=r1)
    return and_op

def get_RRRVfmadd231pdOp(r1, r2, r3):
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)
    r3 = ParserSSAValue(r3)

    vfmadd231pd_op = x86.Vfmadd231pdOp(r1, r2, r3, result=r1)
    return vfmadd231pd_op

def get_MRVmovupdOp(arg1, r2):
    r1, offset = arg1
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)
    offset = IntegerAttr(offset, i64)
    
    vmovupd_op = x86.MRVmovupdOp(r1, r2, offset, result=r1)
    return vmovupd_op

def get_RMVmovupd(r1, arg2):
    r2, offset = arg2
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)
    offset = IntegerAttr(offset, i64)
    
    vmovupd_op = x86.RMVmovupdOp(r1, r2, offset, result=r1)
    return vmovupd_op

def get_RKMVmovupdOp(arg1, arg2):
    r1, mask, modifier = arg1
    r2, offset = arg2
    r1 = ParserSSAValue(r1)
    mask = ParserSSAValue(mask)
    r2 = ParserSSAValue(r2)
    
    vmovupd_op = x86.RKMVmovupdOp(r1, r2, mask, offset, modifier, result=r1)
    return vmovupd_op

def get_MPrefetcht1Op(arg1):
    r1, offset = arg1
    r1 = ParserSSAValue(r1)
    offset = IntegerAttr(offset, i64)
    
    prefetcht1_op = x86.MPrefetcht1Op(r1, offset, result=r1)
    return prefetcht1_op

def get_RMVbroadcastsdOp(r1, arg2):
    r2, offset = arg2
    r1 = ParserSSAValue(r1)
    r2 = ParserSSAValue(r2)
    offset = IntegerAttr(offset, i64)
    
    vbroadcastsd_op = x86.RMVbroadcastsdOp(r1, r2, offset, result=r1)
    return vbroadcastsd_op

def get_MKRVmovupdOp(arg1, r2):
    r1, mask, offset, modifier = arg1
    r1 = ParserSSAValue(r1)
    mask = ParserSSAValue(mask)
    r2 = ParserSSAValue(r2)
    
    vmovupd_op = x86.MKRVmovupdOp(r1, r2, mask, offset, modifier, result=r1)
    return vmovupd_op

def get_RICmpOp(r1, i):
    r1 = ParserSSAValue(r1)
    i = IntegerAttr(i, i64)
    
    cmp_op = x86.RICmpOp(r1, i, result=r1)
    return cmp_op

def get_MJlOp(arg1):
    
    
    
# print(type(ParserSSAValue(x86.Registers.RAX)))
# print(type((x86.Registers.RAX)))



class AssemblyParser:
    def __init__(self):
        self.registers_map = {}
        
        # general purpose registers
        for register in x86.GeneralRegisterType.X86_INDEX_BY_NAME.keys():
            register_object = getattr(x86.Registers, register.upper())
            self.registers_map[register] = register_object
            
        # mask registers
        for register in x86.MaskRegisterType.X86MASK_INDEX_BY_NAME.keys():
            register_object = getattr(x86.Registers, register.upper())
            self.registers_map[register] = register_object
            
        # simd registers
        for register in x86.SIMDRegisterType.X86SIMD_INDEX_BY_NAME.keys():
            register_object = getattr(x86.Registers, register.upper())
            self.registers_map[register] = register_object
        
        
        self.operand_type_map = {
            "Register" : 0,
            "Memory" : 1,
            "Immediate" : 2,
            "Masked Register": 3,
            "Masked Memory": 4,
        }
        
        self.label_map = {}

    def parse_file(self, file_path):
        # Read the entire file to process labels first
        with open(file_path, "r") as file:
            lines = file.readlines()

        block = []
        # process instructions
        for line in lines:
            line = line.split("#")[0].strip()  # Remove comments
            line = line.split(";")[0].strip()  # Remove comments
            if not line:  # Skip empty lines and labels
                continue
            if line.endswith(":"):
                label = line[:-1]  # Remove the colon
                self.label_map[label] = None  # Placeholder for label handling
                continue
            try:
                print(line)
                output = self.parse_instruction(line)
                if output:  # Some instructions might not return an output
                    # print(output)
                    # print(output.assembly_line())
                    block.append(output)
            except ValueError as e:
                print(f"Error parsing line '{line}': {e}")
                
        return block

    def parse_instruction(self, instruction):
        parts = instruction.split(maxsplit=1)
        if len(parts) < 2:
            return None  # No operation to perform, could be a label line

        operation, operands = parts[0], parts[1]
        # print(operation,  ":", operands)
        operands = [element.strip() for element in operands.split(',')]  # Split operands and remove commas
        if PARSER_ASM_SYNTAX == "att":
            operands = operands[::-1]
        
        # Parse operands
        operands = self.parse_operands(operands)
        
        # Handling each operation
        if operation == "mov":
            return self.parse_mov(operation, operands)
        elif operation == "add":
            return self.parse_add(operation, operands)
        elif operation == "push":
            return self.parse_push(operation, operands)
        elif operation == "sub":
            return self.parse_sub(operation, operands)
        elif operation == "movabs":
            return self.parse_movabs(operation, operands)
        elif operation == "and":
            return self.parse_and(operation, operands)
        elif operation == "kmovb":  
            return self.parse_kmovb(operation, operands)
        elif operation == "vfmadd231pd":
            return self.parse_vfmadd231pd(operation, operands)
        elif operation == "vmovupd":
            return self.parse_vmovupd(operation, operands)
        elif operation == "prefetcht1":
            return self.parse_prefetcht1(operation, operands)
        elif operation == "vbroadcastsd":
            return self.parse_vbroadcastsd(operation, operands)
        elif operation == "cmp":
            return self.parse_cmp(operation, operands)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
    def parse_operands(self, operands):
        parsed_operands = [None] * len(operands)
        for i, operand in enumerate(operands):
            # parse register operands
            if operand.startswith('%'):
                if operand.strip().endswith('}'):
                    parsed_operands[i] = (self.operand_type_map["Masked Register"], self.get_masked_register(operand))
                else:    
                    register = operand[1:]
                    if register in self.registers_map:
                        parsed_operands[i] = (self.operand_type_map["Register"], self.registers_map[register])
                    else:
                        raise ValueError(f"Unsupported register: {register}")
            # parse immediate operands
            elif operand.startswith('$'):
                parsed_operands[i] = (self.operand_type_map["Immediate"], self.convert_to_int(operand[1:]))
            # parse memory operands
            elif operand[0].isdigit():
                parsed_operands[i] =  self.get_memory_operand(operand)
            elif operand.startswith("("):
                parsed_operands[i] = self.get_memory_operand(operand)
            else:
                raise ValueError(f"Unsupported operand: {operand}")
        return parsed_operands
    
    def convert_to_int(self, s):
        if s.startswith("0x"):
            return int(s, 16)
        elif s.startswith("0b"):
            return int(s, 2)
        else:
            return int(s, 10)
        
    def get_memory_operand(self, operand):
        offset = 0x0
        
        parts = operand.strip().split('(')
        if len(parts[0]) > 0:
            offset = self.convert_to_int(parts[0])
            
        if parts[1].endswith('}'):
            #0x40(%rdx){%k1}
            subparts = parts[1].split('{')
            register = subparts[0][1:].replace(')', '')
            mask = subparts[1][1:].replace('}', '')
            if len(subparts) > 2:
                modifier = subparts[2].replace('}', '')
            else:
                modifier = None
            if register in self.registers_map and mask in self.registers_map:
                return (self.operand_type_map["Masked Memory"], (self.registers_map[register], self.registers_map[mask], offset, modifier))
            else:
                raise ValueError(f"Unsupported register: {register} or mask: {mask}")
      
        else:
            register = parts[1][1:].replace(')', '') if len(parts) > 1 else None
            if register in self.registers_map:
                return (self.operand_type_map["Memory"], (self.registers_map[register], offset))
            else:
                raise ValueError(f"Unsupported register: {register}")

        
        
        return (register, offset)

    
    def get_masked_register(self, operand):
        # process %zmm23{%k1}{z}
        modifier = None
        parts = operand.split('{')
        register = parts[0][1:]
        mask = parts[1][1:].replace('}', '')
        if len(parts) > 2:
            modifier = parts[2].replace('}', '')
        
        if register in self.registers_map and mask in self.registers_map:
            return (self.registers_map[register], self.registers_map[mask], modifier)
        else:
            raise ValueError(f"Unsupported register: {register} or mask: {mask}")
            
        
        
    def parse_mov(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        
        # print(operation, operands, operand_types)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Register"]):
            return get_RRMovOp(operands[0][1], operands[1][1])   
        elif operand_types == (self.operand_type_map["Register"], self.operand_type_map["Memory"]):
            return get_RMMovOp(operands[0][1], operands[1][1])
        elif operand_types == (self.operand_type_map["Register"], self.operand_type_map["Immediate"]):
            return get_RIMovOp(operands[0][1], operands[1][1])
        elif operand_types == (self.operand_type_map["Memory"], self.operand_type_map["Register"]):
            return get_MRMovOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for mov")

    def parse_add(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Register"]):
            return get_RRAddOp(operands[0][1], operands[1][1])
        elif operand_types == (self.operand_type_map["Register"], self.operand_type_map["Immediate"]):
            return get_RIAddOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for add")
        
    def parse_push(self, operation, operands):
        if len(operands) != 1:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], ):
            return get_RPushOp(operands[0][1])
        else:
            raise ValueError("Unsupported operands for push")
        
    def parse_sub(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Immediate"]):
            return get_RISubOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for sub")
        
    def parse_movabs(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Immediate"]):
            return get_RIMovabsOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for movabs")
        
    def parse_and(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Register"]):
            return get_RRAndOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for and")
        
    def parse_kmovb(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Register"]):
            return get_RRKmovbOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for kmovb")
        
    def parse_vfmadd231pd(self, operation, operands):
        if len(operands) != 3:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Register"], self.operand_type_map["Register"]):
            return get_RRRVfmadd231pdOp(operands[0][1], operands[1][1], operands[2][1])
        else:
            raise ValueError("Unsupported operands for vfmadd231pd")
        
    def parse_vmovupd(self, operation, operands):
        # print(operands)
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Memory"]):
            return get_RMVmovupd(operands[0][1], operands[1][1])
        elif operand_types == (self.operand_type_map["Memory"], self.operand_type_map["Register"]):
            return get_MRVmovupdOp(operands[0][1], operands[1][1])
        elif operand_types == (self.operand_type_map["Masked Register"], self.operand_type_map["Memory"]):
            return get_RKMVmovupdOp(operands[0][1], operands[1][1])
        elif operand_types == (self.operand_type_map["Masked Memory"], self.operand_type_map["Register"]):
            return get_MKRVmovupdOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for vmovupd")
        
    def parse_prefetcht1(self, operation, operands):
        if len(operands) != 1:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Memory"], ):
            return get_MPrefetcht1Op(operands[0][1])
        else:
            raise ValueError("Unsupported operands for prefetcht1")
        
    def parse_vbroadcastsd(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Memory"]):
            return get_RMVbroadcastsdOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for vbroadcastsd")
        
    def parse_cmp(self, operation, operands):
        if len(operands) != 2:
            raise ValueError("Invalid number of operands")
        operand_types = tuple(operand[0] for operand in operands)
        if operand_types == (self.operand_type_map["Register"], self.operand_type_map["Immediate"]):
            return get_RICmpOp(operands[0][1], operands[1][1])
        else:
            raise ValueError("Unsupported operands for cmp")
    
    



parser = AssemblyParser()
block = parser.parse_file("/homes/ajayati/scratch/research/superopt/misc/asm.s")

# for line in block:
#     print(line)

# print(len(block))

for line in block:
    # print(line)
    print(line.assembly_line())

# general_purpose_registers = x86.GeneralRegisterType.X86_INDEX_BY_NAME
# print(general_purpose_registers.keys())