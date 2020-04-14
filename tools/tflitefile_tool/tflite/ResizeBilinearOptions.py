# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers


class ResizeBilinearOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsResizeBilinearOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ResizeBilinearOptions()
        x.Init(buf, n + offset)
        return x

    # ResizeBilinearOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ResizeBilinearOptions
    def AlignCorners(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return bool(
                self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False


def ResizeBilinearOptionsStart(builder):
    builder.StartObject(3)


def ResizeBilinearOptionsAddAlignCorners(builder, alignCorners):
    builder.PrependBoolSlot(2, alignCorners, 0)


def ResizeBilinearOptionsEnd(builder):
    return builder.EndObject()
