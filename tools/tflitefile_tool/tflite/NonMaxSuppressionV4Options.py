# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class NonMaxSuppressionV4Options(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsNonMaxSuppressionV4Options(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NonMaxSuppressionV4Options()
        x.Init(buf, n + offset)
        return x

    # NonMaxSuppressionV4Options
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)


def NonMaxSuppressionV4OptionsStart(builder):
    builder.StartObject(0)


def NonMaxSuppressionV4OptionsEnd(builder):
    return builder.EndObject()
