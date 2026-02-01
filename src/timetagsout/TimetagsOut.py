import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import hashlib 

class TimetagsOut:

    def __init__(self, *args, **kwargs):

        if args:

            self.filepath = args[0]
            
            with h5py.File(args[0],'r') as f:

                Channel = f.get("Channel")
                Channel = np.array(Channel).squeeze()

                Timetag = f.get("ProcessedTimetag")
                Timetag = np.array(Timetag).squeeze()

            self.Channel = Channel
            self.Timetag = Timetag

        else:
            self.filepath = None
            self.Channel = kwargs.get('channel')
            self.Timetag = kwargs.get('timetags')

    
    def preview(self, rows: int = 10):

        if self.filepath == None:

            data = {

                'Channel': self.Channel,
                'ProcessedTimetag': self.Timetag

            }

            print (f"number of rows: {len(self.Timetag)}")

            df = pd.DataFrame(data)

            return df.tail(rows)

        
        else:
            with h5py.File(self.filepath,'r') as f:
                Channel = f.get("Channel")
                Channel = np.array(Channel).squeeze()

                Edge = f.get("Edge")
                Edge = np.array(Edge).squeeze()

                Timetag = f.get("Timetag")
                Timetag = np.array(Timetag).squeeze()

                ProcessedTimetag = f.get("ProcessedTimetag")
                ProcessedTimetag = np.array(ProcessedTimetag).squeeze()

            
            data = {

                'Channel': Channel,
                'Edge': Edge,
                'Timetag': Timetag,
                'ProcessedTimetag': ProcessedTimetag
            }

            print (f"number of rows: {len(ProcessedTimetag)}")

            df = pd.DataFrame(data)

            return df.tail(rows)

        


    def window_extract_bits(
        self,
        ch_1: float,
        ch_2: float,
        window_width: float,
        write_to_file: bool = False
    ):


        channel = self.Channel
        timetag = self.Timetag

        # --- filter to only ch_1 and ch_2 ---
        mask = (channel == ch_1) | (channel == ch_2)
        ch = channel[mask]
        t = timetag[mask]

        if t.size == 0:
            bits = np.array([], dtype=np.uint8)
            self.bit = bits
            if write_to_file:
                print("0 bits are generated. 0 zeros and 0 ones")
            return bits, 0.0

        # --- compute bins ---
        bins = np.floor(t / window_width).astype(np.int64)

        # --- sort by bins (key speed trick) ---
        order = np.argsort(bins, kind="mergesort")  
        bins_s = bins[order]
        ch_s = ch[order]

        # --- run-length encoding via unique + first indices + counts ---
        # unique_bins: distinct bins in sorted order
        # first_idx: start index in bins_s for each unique bin
        unique_bins, first_idx, counts = np.unique(bins_s, return_index=True, return_counts=True)

        # bins with exactly one event:
        keep = (counts == 1)
        single_event_pos = first_idx[keep]              # single event position    
        single_channels = ch_s[single_event_pos]    

        # --- channel -> bit (vectorized) ---
        bits = (single_channels == ch_1).astype(np.uint8)  # ch_1 -> 1, ch_2 -> 0
        self.bit = bits

        # pre-processing algorithm efficiency
        efficiency = bits.size / t.size

        if write_to_file:
            # --- filename ---
            if getattr(self, "filepath", None) is None:
                random_index = np.random.randint(1, 100)
                filename = f"Single_photon_{random_index}.txt"
            else:
                filename = self.filepath.replace(".h5", ".txt")

            with open(filename, "w") as f:
                f.write("".join(bits.astype(str)))

            num0 = int(np.sum(bits == 0))
            num1 = int(np.sum(bits == 1))
            print(f"{bits.size} bits are generated. {num0} zeros and {num1} ones")

        return bits, efficiency
    
    def von_neumann(self, write_to_file: bool = False): 

        if self.bit is None:
            raise ValueError("bits is missing, run window_extract_bits() first")

        if len(self.bit) % 2 == 1:

            bit = self.bit[:-1]

        else:
            bit = self.bit

        b0 = bit[0::2]
        b1 = bit[1::2]

        keep = (b0 != b1)

        vn_bit = b0[keep].astype(np.uint8)

        input_pairs = len(b0)

        efficiency = len(vn_bit) / input_pairs if input_pairs > 0 else 0
    
        if write_to_file:

            # --- Filename ---
            if self.filepath == None:
                random_index = np.random.randint(low = 1, high = 100)
                filename = 'Single_photon_' + str(random_index) +'_vn.txt'

            else: 
                filename = self.filepath.replace('.h5', '_vn.txt')

            with open(filename, 'w') as f:
                f.write(''.join(vn_bit.astype(str)))

            num0 = np.sum(vn_bit == 0)
            num1 = np.sum(vn_bit == 1)

            print(f"{vn_bit.size} bits are generated. {num0} zeros and {num1} ones")

        return vn_bit, efficiency

    
    def concatenate(self, data2, reverse: bool = False):

        fltr_channel1 = self.Channel
        fltr_timetag1 = self.Timetag

        fltr_channel2 = data2.Channel
        fltr_timetag2 = data2.Timetag

        if reverse:
            combine_channel = np.concatenate((fltr_channel2,fltr_channel1))
            combine_timetag = np.concatenate((fltr_timetag2,fltr_timetag1))

        else:
            combine_channel = np.concatenate((fltr_channel1,fltr_channel2))
            combine_timetag = np.concatenate((fltr_timetag1,fltr_timetag2))

        return TimetagsOut(channel = combine_channel, timetag = combine_timetag)
    

    def hashing256(self,
               input_bits: int = 512,
               output_bits: int = 256,
               write_to_file: bool = True):
        
        if self.bit is None:
            raise ValueError('bit is missing, run window_extract_bit() first')


        # Extractor efficiency (constant, independent of min-entropy)
        efficiency = output_bits / input_bits

        num_blocks = len(self.bit) // input_bits
        extracted_bits = []

        for i in range(num_blocks):
            block = self.bit[i * input_bits:(i + 1) * input_bits]

            # Pack bits into bytes
            block_bytes = np.packbits(block)

            # SHA-256 hash
            digest = hashlib.sha256(block_bytes).digest()

            # Convert hash to bits and keep only output_bits
            digest_bits = np.unpackbits(
                np.frombuffer(digest, dtype=np.uint8)
            )[:output_bits]

            extracted_bits.append(digest_bits)

        extracted_bits = np.concatenate(extracted_bits)

        if write_to_file:
            filename = self.filepath.replace('.h5', '_hashed.txt')

            with open(filename, 'w') as f:
                f.write(''.join(extracted_bits.astype(str)))

            num0 = np.sum(extracted_bits == 0)
            num1 = np.sum(extracted_bits == 1)

            print(
                f'{len(extracted_bits)} bits generated '
                f'({num0} zeros, {num1} ones)'
            )

        return extracted_bits, efficiency
    

    def toeplitz_hashing(
        self,
        h_min: float,
        block_size: int = 512,
        margin: int = 128,
        write_to_file: bool = True):

        if self.bit is None:
            raise ValueError('bit is missing, run window_extract_bit() first')


        # Define block parameters
        n = block_size
        m = int(np.floor(n * h_min)) - margin

        if m <= 0:
            raise ValueError("Output size m must be positive.")

        efficiency = m / n

        num_blocks = len(self.bit) // n
        extracted_bits = []

        # Generate Toeplitz seed
        seed = np.random.randint(0, 2, size=n + m - 1, dtype=np.uint8)

        # Toeplitz hashing
        for i in range(num_blocks):
            block = self.bit[i * n:(i + 1) * n]

            out = np.zeros(m, dtype=np.uint8)

            for row in range(m):
                out[row] = np.mod(
                    np.dot(seed[row:row + n], block),
                    2
                )

            extracted_bits.append(out)

        extracted_bits = np.concatenate(extracted_bits)


        # Write output
        if write_to_file:
            filename = self.filepath.replace('.h5', '_toeplitz.txt')

            with open(filename, 'w') as f:
                f.write(''.join(extracted_bits.astype(str)))

            num0 = np.sum(extracted_bits == 0)
            num1 = np.sum(extracted_bits == 1)

            print(
                f"{len(extracted_bits)} bits are generated: "
                f"{num0} zeros and {num1} ones"
            )
            print(f"Toeplitz hashing efficiency = {efficiency:.4f}")

        return extracted_bits, efficiency
    

    def bias_per_block(self, block_size: int = 1_000_000):

        # Computes bias per fixed-size block using self.bit.
        # Bias(block) = | P(1) - 0.5 |

        if not hasattr(self, "bit"):
            raise AttributeError("self.bit not found. Run window_extract_bits(...) first.")

        bits = np.asarray(self.bit, dtype=np.uint8)
        if bits.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

        n = bits.size
        block_id = np.arange(n) // block_size

        ones = np.bincount(block_id, weights=bits)
        sizes = np.bincount(block_id)

        p1 = ones / sizes
        bias = np.abs(p1 - 0.5)

        x = np.arange(bias.size, dtype=np.int64)
        return x, bias

    @staticmethod
    def min_entropy(*bitstream):

        def min_entropy(bit):
            if bit.size == 0:
                raise ValueError('Empty bit array')
            
            p0 = np.mean(bit == 0)
            p1 = np.mean(bit == 1)

            p_max = max(p0,p1)

            return -np.log2(p_max)
        

        h_min = []

        for i in range (len(bitstream)):

            bit = bitstream[i]

            h_min.append(min_entropy(bit))


        return np.array(h_min, dtype=np.float64)
        
        

    @staticmethod
    def exp_decay_fitting(ac, freq:float, detail:bool = False,  **kwargs):

        lags = np.arange(len(ac))
        x_fit = lags[1:]
        y_fit = ac[1:]

        # Sanity check
        assert np.all(np.isfinite(y_fit))


        # function 
        def exp_decay(t, A, tau, C):

            # t : time/lag
            # tau : effective correlation time (how long past detection events influence future bits)  

            return A*np.exp(-t/tau) + C

        # Initialization
        A0 = y_fit[0]
        tau0 = 5
        C0 = 0


        default_p0 = (A0,tau0,C0)

        p0 = kwargs.get('p0',default_p0)


        params, cov = curve_fit(exp_decay,x_fit,y_fit, p0 = p0)

        A, tau, C = params

        sigma = np.sqrt(np.diag(cov))


        # effective correlation time
        dt = 1/freq
        tau_time = tau*dt
        tau_se = sigma[1]*dt

        tau_eff = np.array([tau_time, tau_se])

        if detail:

            print(f"A   = {A:.3e} ± {sigma[0]:.3e}")
            print(f"tau = {tau:.2f} ± {sigma[1]:.2f} lags")
            print(f"C   = {C:.3e} ± {sigma[2]:.3e}")

            dt = 1e-6  # 1 µs
            tau_time = tau * dt
            print(f"effective correlation time ≈ {tau_time*1e6:.2f} µs")


        # smoothen the fitting curve 
        if len(ac) < 1000:
            x_plot = np.linspace(1,len(ac), 1000)

        else: 
            x_plot = x_fit

        return np.array([x_plot, exp_decay(x_plot,*params)]), tau_eff


    @staticmethod
    # Wiener-Kinchin theorem
    def fft_autocorrelation(bits:np.ndarray, max_lag:int = None, fitting:bool = False):

        x = bits

        x = x - np.mean(x)

        # --- zero padding : to prevent large-lag terms wrap around ---
        N = len(x)
        X = np.fft.fft(x, n = 2*N)

        # power spectral density
        S = X*np.conj(X)

        # inverse fourier transform to get correlation
        r = np.fft.ifft(S).real[:N]

        # normalization
        r /= r[0]

        if max_lag is not None:
            # slicing

            r = r[1:max_lag]
            rx = np.arange(max_lag)[1:]

            xy_autocorr = [rx,r]


        if fitting:
            xy_fit = TimetagsOut.exp_decay_fitting(r)

            return xy_fit, xy_autocorr
        
        else:

            return r[1:]
    

    @staticmethod
    # unbiased direct autocorrelation
    def autocorrelation(bit:np.ndarray , freq:float, max_lag:int, fitting = True):

        x = np.asarray(bit,dtype = float)
        x -= np.mean(x)
        N = x.size

        r = np.empty(max_lag)

        for k in range (max_lag):
            r[k] = np.dot(x[:N - k],x[k:N])/(N - k)

        r = r/r[0]


        # fitting
        xy_fitting, tau = TimetagsOut.exp_decay_fitting(r, freq)

        xy_autocorr = [np.arange(max_lag)[1:], r[1:]]

        if fitting:
            return xy_fitting, xy_autocorr, tau
        
        else:
            return xy_autocorr, tau
        










    


        

