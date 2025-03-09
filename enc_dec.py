from decimal import Decimal, getcontext

getcontext().prec = 300  # precision

def enc(value, Lw1, Up1, pb):

    pcum = [Decimal('0')]
    s = sum(pb)  # float
    s_dec = Decimal(str(s))
    run_sum = Decimal('0')
    for x in pb:
        run_sum += Decimal(str(x))
        pcum.append(run_sum)

    total = pcum[-1]
    for i in range(len(pcum)):
        pcum[i] = pcum[i] / total
    
    p1 = pcum[value]
    p2 = pcum[value + 1]

    Lw1 = Decimal(str(Lw1))
    Up1 = Decimal(str(Up1))
    lu = Up1 - Lw1
    Lw = Lw1 + lu * p1
    Up = Lw1 + lu * p2

    cnew_bits = []

    half = Decimal('0.5')
    one  = Decimal('1')
    two  = Decimal('2')

    while (Lw >= half) or (Up <= half):
        if Lw >= half:
            Lw = (Lw * two) - one
            Up = (Up * two) - one
            cnew_bits.append('1')
        else:
            Lw = Lw * two
            Up = Up * two
            cnew_bits.append('0')

    return Lw, Up, ''.join(cnew_bits)

def dcd(Code, Lw1, Up1, Lw2, Up2, pb):

    pcum = [Decimal('0')]
    run_sum = Decimal('0')
    for x in pb:
        run_sum += Decimal(str(x))
        pcum.append(run_sum)

    total = pcum[-1]
    for i in range(len(pcum)):
        pcum[i] = pcum[i] / total
        
    Lw1 = Decimal(str(Lw1))
    Up1 = Decimal(str(Up1))
    Lw2 = Decimal(str(Lw2))
    Up2 = Decimal(str(Up2))

    lu = Up1 - Lw1
    p1 = [Lw1 + lu * pcum[i] for i in range(len(pcum))]

    Ln = len(pb)
    bt = 0
    fd = False
    fst = 0

    vn = None
    half = Decimal('0.5')
    one  = Decimal('1')
    two  = Decimal('2')

    while not fd:
        search_array = p1[fst+1 : Ln+1]
        idxs = [idx for idx, val in enumerate(search_array) if val > Lw2]
        if len(idxs) > 0:
            vn = fst + idxs[0]
        else:
            vn = Ln - 1

        fst = vn - 1

        if (vn < Ln) and (p1[vn+1] >= Up2):
            fd = True
        else:
            if Code[bt] == '1':
                Lw2 = Lw2 + (Up2 - Lw2)/two
            else:
                Up2 = Lw2 + (Up2 - Lw2)/two
            bt += 1

    Lw = p1[vn]
    Up = p1[vn+1]
    Code_rem = Code[bt:]

    while (Lw >= half) or (Up <= half):
        if Lw >= half:
            Lw  = Lw*two - one
            Up  = Up*two - one
            Lw2 = Lw2*two - one
            Up2 = Up2*two - one
        else:
            Lw  = Lw*two
            Up  = Up*two
            Lw2 = Lw2*two
            Up2 = Up2*two

    return vn, Code_rem, Lw, Up, Lw2, Up2