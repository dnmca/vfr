

class KeyPoint:
    """
    Volleyball field model:

    TL          TLA    TLM    TRA          TR
    @------------@------@------@------------@
    |            |      |      |            |
    |            |      |      |            |
    |            |      |      |            |
    |    LEFT    |      |      |    RIGHT   |
    |            |      |      |            |
    |            |      |      |            |
    |            |      |      |            |
    @------------@------@------@------------@
    BL          BLA    BMA    BRA          BR

    """
    TL =  'ToLe'
    TLA = 'ToLeA'
    TM =  'ToMi'
    TRA = 'ToRiA'
    TR =  'ToRi'
    BL =  'BoLe'
    BLA = 'BoLeA'
    BM =  'BoMi'
    BRA = 'BoRiA'
    BR =  'BoRi'


KEYPOINTS = {
    KeyPoint.TL:  {'id': 0, 'x':  0.0, 'y': 0.0},
    KeyPoint.TLA: {'id': 1, 'x':  6.0, 'y': 0.0},
    KeyPoint.TM:  {'id': 2, 'x':  9.0, 'y': 0.0},
    KeyPoint.TRA: {'id': 3, 'x': 12.0, 'y': 0.0},
    KeyPoint.TR:  {'id': 4, 'x': 18.0, 'y': 0.0},
    KeyPoint.BL:  {'id': 5, 'x':  0.0, 'y': 9.0},
    KeyPoint.BLA: {'id': 6, 'x':  6.0, 'y': 9.0},
    KeyPoint.BM:  {'id': 7, 'x':  9.0, 'y': 9.0},
    KeyPoint.BRA: {'id': 8, 'x': 12.0, 'y': 9.0},
    KeyPoint.BR:  {'id': 9, 'x': 18.0, 'y': 9.0}
}

CONNECTIONS = [
    (KeyPoint.TL,  KeyPoint.BL),
    (KeyPoint.TLA, KeyPoint.BLA),
    (KeyPoint.TM,  KeyPoint.BM),
    (KeyPoint.TRA, KeyPoint.BRA),
    (KeyPoint.TR,  KeyPoint.BR),
    (KeyPoint.TL,  KeyPoint.TLA),
    (KeyPoint.TLA, KeyPoint.TM),
    (KeyPoint.TM,  KeyPoint.TRA),
    (KeyPoint.TRA, KeyPoint.TR),
    (KeyPoint.BL,  KeyPoint.BLA),
    (KeyPoint.BLA, KeyPoint.BM),
    (KeyPoint.BM,  KeyPoint.BRA),
    (KeyPoint.BRA, KeyPoint.BR)
]