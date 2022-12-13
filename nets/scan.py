import twain
sm = twain.SourceManager(0)
ss = sm.OpenSource()
ss.RequestAcquire(0,0)
rv = ss.XferImageNatively()
if rv:
    (handle, count) = rv
    twain.DIBToBMFile(handle, 'image.bmp')