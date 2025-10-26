
import sys
sys.path.append('./')
from pycore.tikzeng import *
from pycore.blocks  import *

img_size = 64

wConv = 4
wPool = wRelu = 1

arch = [ 
    to_head('.'), 
    to_cor(),
    to_begin(),
    
    #input
    to_Conv(name='input_layer', s_filer=" ", n_filer=3, offset="(-1,0,0)", to="(0,0,0)", width=0.1, height=img_size*2, depth=img_size*2), # output size w/h = 128
    to_input( './images/image1.png' , name='img_input', to='(-3,-2,-5.1)', width=26, height=25.8, caption="Aerial image shape:(256, 256, 3)"),# output size w/h = 256
    
    #block-001
    to_Conv(name='conv0', s_filer=128, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=wConv, height=img_size, depth=img_size, caption="$Encoder (Resnet34) ->$"), # output size w/h = 128
    to_connection(of="input_layer", to="conv0"),
    to_Relu(name="relu0", offset="(0,0,0)", to="(conv0-east)", width=wRelu, height=img_size, depth=img_size, opacity=0.5), # output size w/h = 64
    to_Pool(name="pooling0", s_filer=64, offset="(0,0,0)", to="(relu0-east)", width=wPool, height=img_size//2, depth=img_size//2, opacity=0.5), # output size w/h = 64
    
    to_Conv(name='stage1_unit1_conv1', s_filer=64, n_filer=64, offset="(3,0,0)", to="(pooling0-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_connection(of="pooling0", to="stage1_unit1_conv1"),
    to_Relu(name="stage1_unit1_relu2", offset="(0,0,0)", to="(stage1_unit1_conv1-east)", width=wRelu, height=img_size//2, depth=img_size//2, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage1_unit1_conv2', s_filer=64, n_filer=64, offset="(0,0,0)", to="(stage1_unit1_relu2-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64

    to_Conv(name='stage1_unit1_sc', s_filer=64, n_filer=64, offset="(3,10,0)", to="(pooling0-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    
    to_Conv(name='stage1_unit2_conv1', s_filer=64, n_filer=64, offset="(1,0,0)", to="(stage1_unit1_conv2-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_connection(of="stage1_unit1_conv2", to="stage1_unit2_conv1"),
    blockmidblock(of=["pooling0", 'stage1_unit1_conv1'], mid='stage1_unit1_sc', to=['stage1_unit1_conv2', "stage1_unit2_conv1"], h=10, shift=1.4),
    to_Relu(name="stage1_unit2_relu2", offset="(0,0,0)", to="(stage1_unit2_conv1-east)", width=wRelu, height=img_size//2, depth=img_size//2, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage1_unit2_conv2', s_filer=64, n_filer=64, offset="(0,0,0)", to="(stage1_unit2_relu2-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    
    to_Conv(name='stage1_unit3_conv1', s_filer=64, n_filer=64, offset="(1,0,0)", to="(stage1_unit2_conv2-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_skip(of="stage1_unit2_conv1", to='stage1_unit3_conv1', pos=[1.25, 1.25], init="", end="west"),
    to_connection(of="stage1_unit2_conv2", to="stage1_unit3_conv1"),
    to_Relu(name="stage1_unit3_relu2", offset="(0,0,0)", to="(stage1_unit3_conv1-east)", width=wRelu, height=img_size//2, depth=img_size//2, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage1_unit3_conv2', s_filer=64, n_filer=64, offset="(0,0,0)", to="(stage1_unit3_relu2-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    
    to_skip(of="stage1_unit3_conv1", to='stage1_unit3_conv2', pos=[1.25, 1.25], init="", end="east"),
    
    to_Conv(name='stage2_unit1_conv1', s_filer=32, n_filer=128, offset="(3,0,0)", to="(stage1_unit3_conv2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_connection(of="stage1_unit3_conv2", to="stage2_unit1_conv1"),
    to_Relu(name="stage2_unit1_relu2", offset="(0,0,0)", to="(stage2_unit1_conv1-east)", width=wRelu, height=img_size//4, depth=img_size//4, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage2_unit1_conv2', s_filer=32, n_filer=128, offset="(0,0,0)", to="(stage2_unit1_relu2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    
    to_Conv(name='stage2_unit1_sc', s_filer=32, n_filer=128, offset="(3,5,0)", to="(stage1_unit3_conv2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    
    to_Conv(name='stage2_unit2_conv1', s_filer=32, n_filer=128, offset="(1,0,0)", to="(stage2_unit1_conv2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_connection(of="stage2_unit1_conv2", to="stage2_unit2_conv1"),
    blockmidblock(of=["stage1_unit3_conv2", 'stage2_unit1_conv1'], mid='stage2_unit1_sc', to=['stage2_unit1_conv2', "stage2_unit2_conv1"], h=5, shift=1.4),
    to_Relu(name="stage2_unit2_relu2", offset="(0,0,0)", to="(stage2_unit2_conv1-east)", width=wRelu, height=img_size//4, depth=img_size//4, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage2_unit2_conv2', s_filer=32, n_filer=128, offset="(0,0,0)", to="(stage2_unit2_relu2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    
    to_Conv(name='stage2_unit3_conv1', s_filer=32, n_filer=128, offset="(1,0,0)", to="(stage2_unit2_conv2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_connection(of="stage2_unit2_conv2", to="stage2_unit3_conv1"),
    to_skip(of="stage2_unit2_conv1", to='stage2_unit3_conv1', pos=[1.25, 1.25], init="", end="west"),
    to_Relu(name="stage2_unit3_relu2", offset="(0,0,0)", to="(stage2_unit3_conv1-east)", width=wRelu, height=img_size//4, depth=img_size//4, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage2_unit3_conv2', s_filer=32, n_filer=128, offset="(0,0,0)", to="(stage2_unit3_relu2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    
    to_Conv(name='stage2_unit4_conv1', s_filer=32, n_filer=128, offset="(1,0,0)", to="(stage2_unit3_conv2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_connection(of="stage2_unit3_conv2", to="stage2_unit4_conv1"),
    to_skip(of="stage2_unit3_conv1", to='stage2_unit4_conv1', pos=[1.25, 1.25], init="", end="west"),
    to_Relu(name="stage2_unit4_relu2", offset="(0,0,0)", to="(stage2_unit4_conv1-east)", width=wRelu, height=img_size//4, depth=img_size//4, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage2_unit4_conv2', s_filer=32, n_filer=128, offset="(0,0,0)", to="(stage2_unit4_relu2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    
    to_skip(of="stage2_unit4_conv1", to='stage2_unit4_conv2', pos=[1.25, 1.25], init="", end="east"),
    
    to_Conv(name='stage3_unit1_conv1', s_filer=16, n_filer=256, offset="(2,0,0)", to="(stage2_unit4_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="stage2_unit4_conv2", to="stage3_unit1_conv1"),
    to_Relu(name="stage3_unit1_relu2", offset="(0,0,0)", to="(stage3_unit1_conv1-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage3_unit1_conv2', s_filer=16, n_filer=256, offset="(0,0,0)", to="(stage3_unit1_relu2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    
    to_Conv(name='stage3_unit1_sc', s_filer=16, n_filer=256, offset="(2,2.5,0)", to="(stage2_unit4_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    
    to_Conv(name='stage3_unit2_conv1', s_filer=16, n_filer=256, offset="(1,0,0)", to="(stage3_unit1_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="stage3_unit1_conv2", to="stage3_unit2_conv1"),
    blockmidblock(of=["stage2_unit4_conv1", 'stage3_unit1_conv1'], mid='stage3_unit1_sc', to=['stage3_unit1_conv2', "stage3_unit2_conv1"], h=2.5, shift=1.4, pos_of=0.6),
    to_Relu(name="stage3_unit2_relu2", offset="(0,0,0)", to="(stage3_unit2_conv1-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage3_unit2_conv2', s_filer=16, n_filer=256, offset="(0,0,0)", to="(stage3_unit2_relu2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    
    to_Conv(name='stage3_unit3_conv1', s_filer=16, n_filer=256, offset="(1,0,0)", to="(stage3_unit2_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="stage3_unit2_conv2", to="stage3_unit3_conv1"),
    to_skip(of="stage3_unit2_conv1", to='stage3_unit3_conv1', pos=[1.5, 1.5], init="", end="west"),
    to_Relu(name="stage3_unit3_relu2", offset="(0,0,0)", to="(stage3_unit3_conv1-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage3_unit3_conv2', s_filer=16, n_filer=256, offset="(0,0,0)", to="(stage3_unit3_relu2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    
    to_Conv(name='stage3_unit4_conv1', s_filer=16, n_filer=256, offset="(1,0,0)", to="(stage3_unit3_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="stage3_unit3_conv2", to="stage3_unit4_conv1"),
    to_skip(of="stage3_unit3_conv1", to='stage3_unit4_conv1', pos=[1.5, 1.5], init="", end="west"),
    to_Relu(name="stage3_unit4_relu2", offset="(0,0,0)", to="(stage3_unit4_conv1-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage3_unit4_conv2', s_filer=16, n_filer=256, offset="(0,0,0)", to="(stage3_unit4_relu2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    
    to_Conv(name='stage3_unit5_conv1', s_filer=16, n_filer=256, offset="(1,0,0)", to="(stage3_unit4_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="stage3_unit4_conv2", to="stage3_unit5_conv1"),
    to_skip(of="stage3_unit4_conv1", to='stage3_unit5_conv1', pos=[1.5, 1.5], init="", end="west"),
    to_Relu(name="stage3_unit5_relu2", offset="(0,0,0)", to="(stage3_unit5_conv1-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage3_unit5_conv2', s_filer=16, n_filer=256, offset="(0,0,0)", to="(stage3_unit5_relu2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    
    to_Conv(name='stage3_unit6_conv1', s_filer=16, n_filer=256, offset="(1,0,0)", to="(stage3_unit5_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="stage3_unit5_conv2", to="stage3_unit6_conv1"),
    to_skip(of="stage3_unit5_conv1", to='stage3_unit6_conv1', pos=[1.5, 1.5], init="", end="west"),
    to_Relu(name="stage3_unit6_relu2", offset="(0,0,0)", to="(stage3_unit6_conv1-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage3_unit6_conv2', s_filer=16, n_filer=256, offset="(0,0,0)", to="(stage3_unit6_relu2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    
    to_skip(of="stage3_unit6_conv1", to='stage3_unit6_conv2', pos=[1.5, 1.5], init="", end="east"),
    
    to_Conv(name='stage4_unit1_conv1', s_filer=8, n_filer=512, offset="(1.5,0,0)", to="(stage3_unit6_conv2-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    to_connection(of="stage2_unit4_conv2", to="stage4_unit1_conv1"),
    to_Relu(name="stage4_unit1_relu2", offset="(0,0,0)", to="(stage4_unit1_conv1-east)", width=wRelu, height=img_size//16, depth=img_size//16, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage4_unit1_conv2', s_filer=8, n_filer=512, offset="(0,0,0)", to="(stage4_unit1_relu2-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    
    to_Conv(name='stage4_unit1_sc', s_filer=8, n_filer=512, offset="(1.5,2.5,0)", to="(stage3_unit6_conv2-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    
    to_Conv(name='stage4_unit2_conv1', s_filer=8, n_filer=512, offset="(1,0,0)", to="(stage4_unit1_conv2-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    to_connection(of="stage4_unit1_conv2", to="stage4_unit2_conv1"),
    blockmidblock(of=["stage3_unit6_conv1", 'stage4_unit1_conv1'], mid='stage4_unit1_sc', to=['stage4_unit1_conv2', "stage4_unit2_conv1"], h=2.5, shift=1.4, pos_of=0.6),
    to_Relu(name="stage4_unit2_relu2", offset="(0,0,0)", to="(stage4_unit2_conv1-east)", width=wRelu, height=img_size//16, depth=img_size//16, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage4_unit2_conv2', s_filer=8, n_filer=512, offset="(0,0,0)", to="(stage4_unit2_relu2-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    
    to_Conv(name='stage4_unit3_conv1', s_filer=8, n_filer=512, offset="(1,0,0)", to="(stage4_unit2_conv2-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    to_connection(of="stage4_unit2_conv2", to="stage4_unit3_conv1"),
    to_skip(of="stage4_unit2_conv1", to='stage4_unit3_conv1', pos=[1.7, 1.75], init="", end="west"),
    to_Relu(name="stage4_unit3_relu2", offset="(0,0,0)", to="(stage4_unit3_conv1-east)", width=wRelu, height=img_size//16, depth=img_size//16, opacity=0.5), # output size w/h = 64
    to_Conv(name='stage4_unit3_conv2', s_filer=8, n_filer=512, offset="(0,0,0)", to="(stage4_unit3_relu2-east)", width=wConv, height=img_size//16, depth=img_size//16, caption="$Bottleneck ->$"), # output size w/h = 64
    
    to_skip(of="stage4_unit3_conv1", to='stage4_unit3_conv2', pos=[1.75, 1.75], init="", end="east"),
    
    to_Pool(name="block4_mxp", s_filer=4, offset="(2,0,0)", to="(stage4_unit3_conv1-east)", width=wPool, height=img_size//32, depth=img_size//32, opacity=0.5), # output size w/h = 64
    to_connection(of="stage4_unit3_conv2", to="block4_mxp"),
    to_Conv(name='btn_conv1', s_filer=4, n_filer=256, offset="(0,0,0)", to="(block4_mxp-east)", width=wConv, height=img_size//32, depth=img_size//32), # output size w/h = 64
    to_Conv(name='btn_conv2', s_filer=4, n_filer=256, offset="(0,0,0)", to="(block4_mxp-east)", width=wConv, height=img_size//32, depth=img_size//32), # output size w/h = 64
    
    to_skip(of="block4_mxp", to='btn_conv2', pos=[2.5, 2.5], init="", end="east"),
    to_skip(of="stage4_unit3_conv2", to='block4_mxp', pos=[2, 3.5], init="east", end="west"),
    
    to_UpConv(name='btn_upconv1', s_filer=8, n_filer=128, offset="(0,0,0)", to="(btn_conv2-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    to_connection(of="btn_conv2", to="btn_upconv1"),
    to_skip(of="block4_mxp", to='btn_upconv1', pos=[4, 2.25], init="west", end="east"),
    
    to_Conv(name='de_block1_conv0', s_filer=8, n_filer=256, offset="(1,0,0)", to="(btn_upconv1-east)", width=wConv, height=img_size//16, depth=img_size//16, caption="$Decoder ->$"), # output size w/h = 64
    to_connection(of="btn_upconv1", to="de_block1_conv0"),
    to_Relu(name="de_block1_relu0", offset="(0,0,0)", to="(de_block1_conv0-east)", width=wRelu, height=img_size//16, depth=img_size//16, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block1_conv1', s_filer=8, n_filer=128, offset="(0,0,0)", to="(de_block1_relu0-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    to_Relu(name="de_block1_relu1", offset="(0,0,0)", to="(de_block1_conv1-east)", width=wRelu, height=img_size//16, depth=img_size//16, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block1_conv2', s_filer=8, n_filer=256, offset="(0,0,0)", to="(de_block1_relu1-east)", width=wConv, height=img_size//16, depth=img_size//16), # output size w/h = 64
    to_Relu(name="de_block1_relu2", offset="(0,0,0)", to="(de_block1_conv2-east)", width=wRelu, height=img_size//16, depth=img_size//16, opacity=0.5), # output size w/h = 64
    
    to_UpConv(name='de_block1_upconv1', s_filer=16, n_filer=256, offset="(1,0,0)", to="(de_block1_relu2-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="de_block1_relu2", to="de_block1_upconv1"),
    to_skip(of="btn_upconv1", to='de_block1_upconv1', pos=[3, 1.75], init="east", end="west"),
    to_skip(of="stage4_unit1_conv1", to='de_block1_upconv1', pos=[3, 1.75], init="west", end="east", orient="south"),
    
    to_Conv(name='de_block2_conv0', s_filer=16, n_filer=128, offset="(1,0,0)", to="(de_block1_upconv1-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="de_block1_upconv1", to="de_block2_conv0"),
    to_Relu(name="de_block2_relu0", offset="(0,0,0)", to="(de_block2_conv0-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block2_conv1', s_filer=16, n_filer=64, offset="(0,0,0)", to="(de_block2_relu0-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_Relu(name="de_block2_relu1", offset="(0,0,0)", to="(de_block2_conv1-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block2_conv2', s_filer=16, n_filer=128, offset="(0,0,0)", to="(de_block2_relu1-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_Relu(name="de_block2_relu2", offset="(0,0,0)", to="(de_block2_conv2-east)", width=wRelu, height=img_size//8, depth=img_size//8, opacity=0.5), # output size w/h = 64
    
    to_UpConv(name='de_block2_upconv1', s_filer=32, n_filer=128, offset="(1,0,0)", to="(de_block2_relu2-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_connection(of="de_block2_relu2", to="de_block2_upconv1"),
    to_skip(of="de_block1_upconv1", to='de_block2_upconv1', pos=[2.5, 1.5], init="east", end="west"),
    to_skip(of="stage2_unit4_conv2", to='de_block2_upconv1', pos=[1.75, 1.75], init="west", end="east", orient="south"),
    
    to_Conv(name='de_block3_conv0', s_filer=32, n_filer=64, offset="(1,0,0)", to="(de_block2_upconv1-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_connection(of="de_block2_upconv1", to="de_block3_conv0"),
    to_Relu(name="de_block3_relu0", offset="(0,0,0)", to="(de_block3_conv0-east)", width=wRelu, height=img_size//4, depth=img_size//4, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block3_conv1', s_filer=32, n_filer=32, offset="(0,0,0)", to="(de_block3_relu0-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_Relu(name="de_block3_relu1", offset="(0,0,0)", to="(de_block3_conv1-east)", width=wRelu, height=img_size//4, depth=img_size//4, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block3_conv2', s_filer=32, n_filer=64, offset="(0,0,0)", to="(de_block3_relu1-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_Relu(name="de_block3_relu2", offset="(0,0,0)", to="(de_block3_conv2-east)", width=wRelu, height=img_size//4, depth=img_size//4, opacity=0.5), # output size w/h = 64
    
    to_UpConv(name='de_block3_upconv1', s_filer=64, n_filer=64, offset="(1,0,0)", to="(de_block3_relu2-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_connection(of="de_block3_relu2", to="de_block3_upconv1"),
    to_skip(of="stage1_unit3_conv2", to='de_block3_upconv1', pos=[1.75, 1.75], init="west", end="east", orient="south"),
    to_skip(of="de_block2_upconv1", to='de_block3_upconv1', pos=[2, 1.25], init="east", end="west"),
    
    to_Conv(name='de_block4_conv0', s_filer=64, n_filer=32, offset="(1,0,0)", to="(de_block3_upconv1-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_connection(of="de_block3_upconv1", to="de_block4_conv0"),
    to_Relu(name="de_block4_relu0", offset="(0,0,0)", to="(de_block4_conv0-east)", width=wRelu, height=img_size//2, depth=img_size//2, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block4_conv1', s_filer=64, n_filer=16, offset="(0,0,0)", to="(de_block4_relu0-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_Relu(name="de_block4_relu1", offset="(0,0,0)", to="(de_block4_conv1-east)", width=wRelu, height=img_size//2, depth=img_size//2, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block4_conv2', s_filer=64, n_filer=32, offset="(0,0,0)", to="(de_block4_relu1-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_Relu(name="de_block4_relu2", offset="(0,0,0)", to="(de_block4_conv2-east)", width=wRelu, height=img_size//2, depth=img_size//2, opacity=0.5), # output size w/h = 64
    
    to_UpConv(name='de_block4_upconv1', s_filer=128, n_filer=32, offset="(1,0,0)", to="(de_block4_relu2-east)", width=wConv, height=img_size, depth=img_size), # output size w/h = 64
    to_connection(of="de_block3_relu2", to="de_block4_upconv1"),
    to_skip(of="conv0", to='de_block4_upconv1', pos=[1.25, 1.25], init="west", end="east", orient="south"),
    to_skip(of="de_block3_upconv1", to='de_block4_upconv1', pos=[1.75, 1.125], init="east", end="west"),
    
    to_Conv(name='de_block5_conv0', s_filer=128, n_filer=32, offset="(1,0,0)", to="(de_block4_upconv1-east)", width=wConv, height=img_size, depth=img_size), # output size w/h = 64
    to_connection(of="de_block4_upconv1", to="de_block5_conv0"),
    to_Relu(name="de_block5_relu0", offset="(0,0,0)", to="(de_block5_conv0-east)", width=wRelu, height=img_size, depth=img_size, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block5_conv1', s_filer=128, n_filer=16, offset="(0,0,0)", to="(de_block5_relu0-east)", width=wConv, height=img_size, depth=img_size), # output size w/h = 64
    to_Relu(name="de_block5_relu1", offset="(0,0,0)", to="(de_block5_conv1-east)", width=wRelu, height=img_size, depth=img_size, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block5_conv2', s_filer=128, n_filer=32, offset="(0,0,0)", to="(de_block5_relu1-east)", width=wConv, height=img_size, depth=img_size), # output size w/h = 64
    to_Relu(name="de_block5_relu2", offset="(0,0,0)", to="(de_block5_conv2-east)", width=wRelu, height=img_size, depth=img_size, opacity=0.5), # output size w/h = 64
    
    to_UpConv(name='de_block5_upconv1', s_filer=256, n_filer=32, offset="(1,0,0)", to="(de_block5_relu2-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 64
    to_connection(of="de_block4_relu2", to="de_block5_upconv1"),
    to_skip(of="input_layer", to='de_block5_upconv1', pos=[1.25, 1.25], init="west", end="east", orient="south"),
    to_skip(of="de_block4_upconv1", to='de_block5_upconv1', pos=[1.75, 1.125], init="east", end="west"),
    
    to_Conv(name='de_block6_conv0', s_filer=256, n_filer=16, offset="(1,0,0)", to="(de_block5_upconv1-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 64
    to_connection(of="de_block5_upconv1", to="de_block6_conv0"),
    to_Relu(name="de_block6_relu0", offset="(0,0,0)", to="(de_block6_conv0-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block6_conv1', s_filer=256, n_filer=8, offset="(0,0,0)", to="(de_block6_relu0-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 64
    to_Relu(name="de_block6_relu1", offset="(0,0,0)", to="(de_block6_conv1-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    to_Conv(name='de_block6_conv2', s_filer=256, n_filer=16, offset="(0,0,0)", to="(de_block6_relu1-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 64
    to_Relu(name="de_block6_relu2", offset="(0,0,0)", to="(de_block6_conv2-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    
    to_Conv(name='de_block7_conv1', s_filer=256, n_filer=8, offset="(1,0,0)", to="(de_block6_relu2-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 64
    to_Relu(name="de_block7_relu0", offset="(0,0,0)", to="(de_block7_conv1-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    
    to_Conv(name='output_layer0', s_filer=" ", n_filer=1, offset="(0.25,0,0)", to="(de_block7_relu0-east)", width=0.1, height=img_size*2, depth=img_size*2), # output size w/h = 128
    to_input( './images/build1.png' , name='img_output0', to='(output_layer0-east)', width=26, height=25.8),# output size w/h = 256
    
    to_Conv(name='output_layer1', s_filer=" ", n_filer=1, offset="(0.25,0,0)", to="(output_layer0-east)", width=0.1, height=img_size*2, depth=img_size*2), # output size w/h = 128
    to_input( './images/build1.png' , name='img_output1', to='(output_layer1-east)', width=26, height=25.8),# output size w/h = 256
    
    to_Conv(name='output_layer2', s_filer=" ", n_filer=1, offset="(0.25,0,0)", to="(output_layer1-east)", width=0.1, height=img_size*2, depth=img_size*2), # output size w/h = 128
    to_input( './images/env1.png' , name='img_output2', to='(output_layer2-east)', width=26, height=25.8),# output size w/h = 256
    
    to_Conv(name='output_layer3', s_filer=" ", n_filer=1, offset="(0.25,0,0)", to="(output_layer2-east)", width=0.1, height=img_size*2, depth=img_size*2), # output size w/h = 128
    to_input( './images/road1.png' , name='img_output3', to='(output_layer3-east)', width=26, height=25.8, caption="Output shape of segmentation mask:(256, 256, 3)"),# output size w/h = 256
    
    
    
    to_Conv(name='enc_edge_conv4', s_filer=256, n_filer=1, offset="(0,0,0)", to="(10, 40, 0)", width=0.5, height=img_size*2, depth=img_size*2), # output size w/h = 128¿
    to_input( './images/edge1.png' , name='img_output_edge_enc', to='(enc_edge_conv4-east)', width=26, height=25.8, caption="Output shape of edge encoder detector:(256, 256, 1)"),# output size w/h = 256
    
    to_Relu(name="enc_edge_relu2", offset="(4,0,0)", to="(enc_edge_conv4-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    to_connection(of="enc_edge_relu2", to="enc_edge_conv4", init="west", end="east"),
    to_Conv(name='enc_edge_conv2', s_filer=256, n_filer=128, offset="(0,0,0)", to="(enc_edge_relu2-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 128¿
    to_Relu(name="enc_edge_relu1", offset="(0,0,0)", to="(enc_edge_conv2-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    to_Conv(name='enc_edge_conv1', s_filer=256, n_filer=128, offset="(0,0,0)", to="(enc_edge_relu1-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 128¿
    
    to_UpConv(name='enc_edge_block1_upconv1', s_filer=256, n_filer=32, offset="(4,0,0)", to="(enc_edge_conv1-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 64
    to_connection(of="enc_edge_block1_upconv1", to="enc_edge_conv1", init="west", end="east"),
    
    to_UpConv(name='enc_edge_block2_upconv1', s_filer=128, n_filer=32, offset="(4,0,0)", to="(enc_edge_block1_upconv1-east)", width=wConv, height=img_size, depth=img_size), # output size w/h = 64
    to_connection(of="enc_edge_block2_upconv1", to="enc_edge_block1_upconv1", init="west", end="east"),
    
    to_UpConv(name='enc_edge_block3_upconv1', s_filer=64, n_filer=64, offset="(4,0,0)", to="(enc_edge_block2_upconv1-east)", width=wConv, height=img_size//2, depth=img_size//2), # output size w/h = 64
    to_connection(of="enc_edge_block3_upconv1", to="enc_edge_block2_upconv1", init="west", end="east"),
    
    to_UpConv(name='enc_edge_block4_upconv1', s_filer=32, n_filer=64, offset="(4,0,0)", to="(enc_edge_block3_upconv1-east)", width=wConv, height=img_size//4, depth=img_size//4), # output size w/h = 64
    to_connection(of="enc_edge_block4_upconv1", to="enc_edge_block3_upconv1", init="west", end="east"),
    to_UpConv(name='enc_edge_block5_upconv1', s_filer=16, n_filer=128, offset="(4,0,0)", to="(enc_edge_block4_upconv1-east)", width=wConv, height=img_size//8, depth=img_size//8), # output size w/h = 64
    to_connection(of="enc_edge_block5_upconv1", to="enc_edge_block4_upconv1", init="west", end="east"),
    to_skip_edge(of=["stage4_unit3_conv2", "block4_mxp"], to=['enc_edge_block5_upconv1'], h=40, shift=0, to_block=True),
    to_skip_edge(of=["stage3_unit6_conv2", "stage4_unit1_conv1"], to=['enc_edge_block5_upconv1', "enc_edge_block4_upconv1"], h=20, shift=-5.35),
    to_skip_edge(of=["stage2_unit4_conv2", "stage3_unit1_conv1"], to=['enc_edge_block4_upconv1', "enc_edge_block3_upconv1"], h=10, shift=7.45),
    to_skip_edge(of=["stage1_unit3_conv2", "stage2_unit1_conv1"], to=['enc_edge_block3_upconv1', "enc_edge_block2_upconv1"], h=15, shift=15.45),
    to_skip_edge(of=["conv0", "pooling0"], to=['enc_edge_block1_upconv1', "enc_edge_block2_upconv1"], h=20, shift=21.6),
    
    
    
    to_UpConv(name='dec_edge_block1_upconv1', s_filer=16, n_filer=128, offset="(2,40,0)", to="(de_block1_conv2-east)", width=wConv, height=img_size//8, depth=img_size//8),
    to_UpConv(name='dec_edge_block2_upconv1', s_filer=32, n_filer=64, offset="(4,0,0)", to="(dec_edge_block1_upconv1-east)", width=wConv, height=img_size//4, depth=img_size//4),
    to_connection(of="dec_edge_block1_upconv1", to="dec_edge_block2_upconv1", init="west", end="east"),
    
    to_UpConv(name='dec_edge_block3_upconv1', s_filer=64, n_filer=64, offset="(4,0,0)", to="(dec_edge_block2_upconv1-east)", width=wConv, height=img_size//2, depth=img_size//2),
    to_connection(of="dec_edge_block2_upconv1", to="dec_edge_block3_upconv1", init="west", end="east"),
    
    to_UpConv(name='dec_edge_block4_upconv1', s_filer=128, n_filer=32, offset="(4,0,0)", to="(dec_edge_block3_upconv1-east)", width=wConv, height=img_size, depth=img_size),
    to_connection(of="dec_edge_block3_upconv1", to="dec_edge_block4_upconv1", init="west", end="east"),
    
    to_UpConv(name='dec_edge_block5_upconv1', s_filer=256, n_filer=32, offset="(4,0,0)", to="(dec_edge_block4_upconv1-east)", width=wConv, height=img_size*2, depth=img_size*2),
    to_connection(of="dec_edge_block4_upconv1", to="dec_edge_block5_upconv1", init="west", end="east"),
    
    to_Conv(name='dec_edge_conv1', s_filer=256, n_filer=128, offset="(4,0,0)", to="(dec_edge_block5_upconv1-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 128¿
    to_connection(of="dec_edge_block5_upconv1", to="dec_edge_conv1", init="west", end="east"),
    to_Relu(name="dec_edge_relu1", offset="(0,0,0)", to="(dec_edge_conv1-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    to_Conv(name='dec_edge_conv2', s_filer=256, n_filer=128, offset="(0,0,0)", to="(dec_edge_relu1-east)", width=wConv, height=img_size*2, depth=img_size*2), # output size w/h = 128¿
    to_Relu(name="dec_edge_relu2", offset="(0,0,0)", to="(dec_edge_conv2-east)", width=wRelu, height=img_size*2, depth=img_size*2, opacity=0.5), # output size w/h = 64
    
    to_Conv(name='dec_edge_conv4', s_filer=256, n_filer=1, offset="(4,0,0)", to="(dec_edge_relu2-east)", width=0.5, height=img_size*2, depth=img_size*2), # output size w/h = 128¿
    to_input( './images/edge1.png' , name='img_output_edge_enc', to='(dec_edge_conv4-east)', width=26, height=25.8, caption="Output shape of edge decoder detector:(256, 256, 1)"),# output size w/h = 256
    to_connection(of="dec_edge_relu2", to="dec_edge_conv4", init="west", end="east"),
    
    to_skip_edge(of=["de_block1_conv2", "de_block1_upconv1"], to=['dec_edge_block1_upconv1'], h=40, shift=0, to_block=True),
    to_skip_edge(of=["de_block2_conv2", "de_block2_upconv1"], to=['dec_edge_block1_upconv1', 'dec_edge_block2_upconv1'], h=5, shift=-1.9),
    to_skip_edge(of=["de_block3_conv2", "de_block3_upconv1"], to=['dec_edge_block2_upconv1', 'dec_edge_block3_upconv1'], h=10, shift=-2.9),
    to_skip_edge(of=["de_block4_conv2", "de_block4_upconv1"], to=['dec_edge_block3_upconv1', 'dec_edge_block4_upconv1'], h=15, shift=-3.9),
    to_skip_edge(of=["de_block5_conv2", "de_block5_upconv1"], to=['dec_edge_block4_upconv1', 'dec_edge_block5_upconv1'], h=17.5, shift=-4.9),
    to_skip_edge(of=["de_block6_conv2", "de_block7_conv1"], to=['dec_edge_block5_upconv1', 'dec_edge_conv1'], h=20, shift=-5.9),

    to_end() 
    ]


def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()
    
