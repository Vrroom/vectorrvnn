# python3 createDBScript.py <svgDir> <outScript>
import os 
import os.path as osp
import sys

if __name__ == "__main__" : 
    createTable = \
        """
CREATE TABLE vectorImages
    (id integer,
     svg xml, 
     primary key (id));

        """
    path = sys.argv[1]
    outScript = sys.argv[2]
    with open(outScript, 'w+') as ofd : 
        ofd.write(createTable)
        for i, file in enumerate(os.listdir(path)) : 
            fullPath = osp.join(path, file)
            with open(fullPath) as ifd :
                content = ifd.read()
            insert = f'INSERT into vectorImages values ({i} , XMLPARSE (DOCUMENT \'{content}\'));\n'
            ofd.write(insert)
