
import os

def to_head( projectpath ):
    pathlayers = os.path.join( projectpath, 'layers/' ).replace('\\', '/')
    return r"""
\documentclass[border=8pt, multi, tikz]{standalone} 
\usepackage{import}
\usepackage{caption}
\subimport{"""+ pathlayers + r"""}{init}
\usetikzlibrary{positioning}
\usetikzlibrary{3d} %for including external image 
"""

def to_cor():
    return r"""
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\AntiPoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}   
\def\SumColor{rgb:blue,5;green,15}
"""

def to_begin():
    return r"""
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}

\begin{document}
\begin{tikzpicture}
\Large\textbf{My title}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
"""

# layers definition

def to_input( pathfile, to='(-3,0,0)', width=8, height=8, name="temp", caption=" "):
    return r"""
\node[canvas is zy plane at x=0] (""" + name + """) at """+ to +""" {
\includegraphics[width="""+ str(width)+"cm"+""",height="""+ str(height)+"cm"+"""]{"""+ pathfile +r"""},
};
\node[canvas is zy plane at x=0, xscale=-1, inner sep=0pt,above=\abovecaptionskip of """ + name + """,text width=\linewidth]
    {"""+caption+r"""};
"""

# Conv
def to_Conv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\ConvColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_UpConv( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +r""",
        xlabel={{"""+ str(n_filer) +""", }},
        zlabel="""+ str(s_filer) +""",
        fill=\AntiPoolColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# Pool
def to_Pool(name, s_filer, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
        fill=\PoolColor,
        zlabel="""+ str(s_filer) +""",
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Relu(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+name+""",
        caption="""+ caption +r""",
        fill=\ConvReluColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

# unpool4, 
def to_UnPool(name, offset="(0,0,0)", to="(0,0,0)", width=1, height=32, depth=32, opacity=0.5, caption=" "):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {Box={
        name="""+ name +r""",
        caption="""+ caption +r""",
        fill=\AntiPoolColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""



def to_ConvRes( name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=6, height=40, depth=40, opacity=0.2, caption=" " ):
    return r"""
\pic[shift={ """+ offset +""" }] at """+ to +""" 
    {RightBandedBox={
        name="""+ name + """,
        caption="""+ caption + """,
        xlabel={{ """+ str(n_filer) + """, }},
        zlabel="""+ str(s_filer) +r""",
        fill={rgb:white,1;black,3},
        bandfill={rgb:white,1;black,2},
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# ConvSoftMax
def to_ConvSoftMax( name, s_filer=40, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""


# SoftMax
def to_SoftMax( name, s_filer=10, offset="(0,0,0)", to="(0,0,0)", width=1.5, height=3, depth=25, opacity=0.8, caption=" " ):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption="""+ caption +""",
        xlabel={{" ","dummy"}},
        zlabel="""+ str(s_filer) +""",
        fill=\SoftmaxColor,
        opacity="""+ str(opacity) +""",
        height="""+ str(height) +""",
        width="""+ str(width) +""",
        depth="""+ str(depth) +"""
        }
    };
"""

def to_Sum( name, offset="(0,0,0)", to="(0,0,0)", radius=2.5, opacity=0.6):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Ball={
        name=""" + name +""",
        fill=\SumColor,
        opacity="""+ str(opacity) +""",
        radius="""+ str(radius) +""",
        logo=$+$
        }
    };
"""


def to_connection( of, to, init="east", end="west"):
    return r"""
\draw [connection]  ("""+of+"""-"""+init+""")    -- node {\midarrow} ("""+to+"""-"""+end+""");
"""

def to_skip( of, to, pos=1.25, init="east", end="", orient="north"):
    or_end = orient
    if orient == "north":
        or_init = "south"
    elif orient == "center":
        or_init = or_end = ""
    else:
        or_init = "north"
    
    return r"""
\path ("""+ of +"""-"""+ or_init + init +""") -- ("""+ of +"""-"""+ or_end + init +""") coordinate[pos="""+ str(pos[0]) +"""] ("""+ of +"""-top) ;
\path ("""+ to +"""-"""+ or_init + end +""")  -- ("""+ to +"""-"""+ or_end + end +""")  coordinate[pos="""+ str(pos[1]) +"""] ("""+ to +"""-top) ;
\draw [connection]  ("""+of+"""-"""+ or_end + init +""")  
-- node {\midarrow}("""+of+"""-top)
-- node {\midarrow}("""+to+"""-top)
-- node {\midarrow} ("""+to+"""-"""+ or_end + end +""");
"""

def blockmidblock( of, mid, to, h=6, shift=2, pos_of=0.4, pos_to=0.4):
    return r"""
\path ("""+ of[0] +"""-east) -- ("""+ of[1] +"""-west) coordinate[pos="""+str(pos_of)+"""] (after1);
\path ("""+ to[0] +"""-east) -- ("""+ to[1] +"""-west) coordinate[pos="""+str(pos_to)+"""] (before1);
\draw [connection]  (after1)  -- node {\midarrow} ++(0,"""+str(h)+""",0) -- node {\midarrow} ("""+mid+"""-west);
\draw [connection]  ("""+mid+"""-east)  -- node {\midarrow} ++("""+str(shift)+""",0,0) -- node {\midarrow} (before1);
"""

def to_skip_edge( of, to, h=6, pos_of=0.4, pos_to=0.4, to_block=False, shift=10):
    latex = r"""\path ("""+ of[0] +"""-east) -- ("""+ of[1] +"""-west) coordinate[pos="""+str(pos_of)+"""] (after1);"""
    if not to_block:
        latex += r"""\path ("""+ to[0] +"""-east) -- ("""+ to[1] +"""-west) coordinate[pos="""+str(pos_to)+"""] (before1);
        \draw [connection]  (after1)  
        -- node {\midarrow} ++(0,"""+str(h)+""",0)  
        -- node {\midarrow} ++("""+str(shift)+""",0,0)  
        -- node {\midarrow} (before1);
        """
    else:
        latex += r"""\draw [connection]  (after1)  
        -- node {\midarrow} ++(0,"""+str(h)+""",0)  
        -- node {\midarrow} ++("""+str(shift)+""",0,0)  
        -- node {\midarrow} ("""+to[0]+"""-east);
        """
    return latex

def to_end():
    return r"""
\node[above,font=\fontsize{70pt}{70pt}\bfseries,] at (current bounding box.north) {U-NET + ERN by Alexander D. Rios};
\end{tikzpicture}
\end{document}
"""


def to_generate( arch, pathname="file.tex" ):
    with open(pathname, "w") as f: 
        for c in arch:
            print(c)
            f.write( c )
     


