# automatically generated by the FlatBuffers compiler, do not modify

# namespace: circle

import flatbuffers

class MatrixDiagOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsMatrixDiagOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = MatrixDiagOptions()
        x.Init(buf, n + offset)
        return x

    # MatrixDiagOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def MatrixDiagOptionsStart(builder): builder.StartObject(0)
def MatrixDiagOptionsEnd(builder): return builder.EndObject()
