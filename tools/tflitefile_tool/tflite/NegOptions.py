# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class NegOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNegOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NegOptions()
        x.Init(buf, n + offset)
        return x

    # NegOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def NegOptionsStart(builder):
    builder.StartObject(0)


def NegOptionsEnd(builder):
    return builder.EndObject()
