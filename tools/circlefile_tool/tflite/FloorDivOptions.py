# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers

class FloorDivOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsFloorDivOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = FloorDivOptions()
        x.Init(buf, n + offset)
        return x

    # FloorDivOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def FloorDivOptionsStart(builder): builder.StartObject(0)
def FloorDivOptionsEnd(builder): return builder.EndObject()
