# Class to read inputs


class DiffusionOpts1D:
    
###############################################################

    def __init__(self):
        self.length = -1

###############################################################

    def read(self, filename):
        inpFile = open(filename, 'r')
        
        for line in inpFile:  

            # Remove trailing white space
            line = line.strip()
            # Remove newline characters
            line = line.strip('\n')
            # Remove string after comment character (#)
            line, scratch1, scratch2 = line.partition('#')
            # Skip empty lines
            if len(line) == 0:
                continue

            keyword, arguments = line.split(' ', 1)
            if keyword == 'length':
                self.length = float(arguments)
                
            elif keyword == 'numgroups':
                self.numGroups = int(arguments)
                
            elif keyword == 'numBins':
                self.numBins = int(arguments)

            elif keyword == 'FisConvergeError':
                self.FisConvError = float(arguments)

            elif keyword == 'FluxConvergeError':
                self.FluxConvError = float(arguments)

            elif keyword == 'ReactorPower':
                self.ReactorPower = float(arguments)

            elif keyword == 'dt':
                self.dt = float(arguments)

            elif keyword == 'subStep':
                self.subStep = float(arguments)

            elif keyword == 't_end':
                self.t_end = float(arguments)

            elif keyword == 'problem':
                self.pb_num = int(arguments)

            elif keyword == 'flux':
                self.flux = str(arguments)

            elif keyword == 'method':
                self.method = str(arguments)

            elif keyword == 'repeat_inner':
                self.repeat_inner = int(arguments)
                
            else:
                continue

        self.delta = self.length/self.numBins

#end class