# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers

class AddNOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsAddNOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = AddNOptions()
        x.Init(buf, n + offset)
        return x

    # AddNOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def AddNOptionsStart(builder): builder.StartObject(0)
def AddNOptionsEnd(builder): return builder.EndObject()
