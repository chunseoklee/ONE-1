# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class ReverseV2Options(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsReverseV2Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReverseV2Options()
        x.Init(buf, n + offset)
        return x

    # ReverseV2Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def ReverseV2OptionsStart(builder):
    builder.StartObject(0)


def ReverseV2OptionsEnd(builder):
    return builder.EndObject()
