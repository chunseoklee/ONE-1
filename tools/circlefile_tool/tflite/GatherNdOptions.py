# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers

class GatherNdOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsGatherNdOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = GatherNdOptions()
        x.Init(buf, n + offset)
        return x

    # GatherNdOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def GatherNdOptionsStart(builder): builder.StartObject(0)
def GatherNdOptionsEnd(builder): return builder.EndObject()
