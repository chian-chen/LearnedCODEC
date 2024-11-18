crfs = [20, 23, 26, 29]

if __name__ == "__main__":

    for crf in crfs:
        with open(f'result{crf}.txt') as f:
            lines = f.readlines()

        bpp = []

        for l in lines:             
            if "bpp" in l and l[4:7] != 'nan':
                bpp.append(round(float(l[4:]), 6))

        print(f'final bpp for {crf}:')
        print(bpp)
