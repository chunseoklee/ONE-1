# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers

class MulOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMulOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MulOptions()
        x.Init(buf, n + offset)
        return x

    # MulOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # MulOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

def MulOptionsStart(builder): builder.StartObject(1)
def MulOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(0, fusedActivationFunction, 0)
def MulOptionsEnd(builder): return builder.EndObject()
