# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers

class BCQGatherOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsBCQGatherOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BCQGatherOptions()
        x.Init(buf, n + offset)
        return x

    # BCQGatherOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BCQGatherOptions
    def InputHiddenSize(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # BCQGatherOptions
    def Axis(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def BCQGatherOptionsStart(builder): builder.StartObject(2)
def BCQGatherOptionsAddInputHiddenSize(builder, inputHiddenSize): builder.PrependInt32Slot(0, inputHiddenSize, 0)
def BCQGatherOptionsAddAxis(builder, axis): builder.PrependInt32Slot(1, axis, 0)
def BCQGatherOptionsEnd(builder): return builder.EndObject()