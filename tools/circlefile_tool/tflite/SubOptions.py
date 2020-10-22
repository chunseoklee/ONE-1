# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers

class SubOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsSubOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = SubOptions()
        x.Init(buf, n + offset)
        return x

    # SubOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # SubOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def SubOptionsStart(builder): builder.StartObject(1)
def SubOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(0, fusedActivationFunction, 0)
def SubOptionsEnd(builder): return builder.EndObject()
