# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class ScatterNdOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsScatterNdOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ScatterNdOptions()
        x.Init(buf, n + offset)
        return x

    # ScatterNdOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def ScatterNdOptionsStart(builder):
    builder.StartObject(0)


def ScatterNdOptionsEnd(builder):
    return builder.EndObject()
