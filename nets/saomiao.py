import twain_module
 
twain_module.acquire(r'./test.bmp',dpi=300,pixel_type="color") # 设置dpi300,彩色模式