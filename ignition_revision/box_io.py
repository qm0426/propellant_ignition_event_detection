import numpy as np

def read_pixel_boxes(path):
    boxes=[];invalid=[]
    if not path.is_file():return boxes,invalid
    for lineno,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        fields=line.split()
        if not fields or fields[0].lower()=='label':continue
        if len(fields) not in (5,6):raise ValueError(f'Unsupported pixel TXT at {path}:{lineno}')
        try:x1,y1,x2,y2=map(float,fields[1:5])
        except ValueError as e:raise ValueError(f'Invalid coordinates at {path}:{lineno}') from e
        row={'source_label':fields[0],'x1':x1,'y1':y1,'x2':x2,'y2':y2,'line':lineno}
        if not np.isfinite([x1,y1,x2,y2]).all() or x2<=x1 or y2<=y1:invalid.append(row);continue
        boxes.append(row)
    return boxes,invalid
