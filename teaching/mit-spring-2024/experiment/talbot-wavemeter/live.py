import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)
ret, frame = cam.read()
cv2.namedWindow("talbot", cv2.WINDOW_NORMAL)
cv2.resizeWindow("talbot", 1600, 900)

# guess parameters
# lengths in microns
pix = 1.38 # pixel pitch
gratinglen = 1.2 # periodicity of grating
theta = 21 * np.pi / 180 # sensor tilt away from grating plane

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
rows, cols = gray.shape
pad_factor = 1
FFTsize = cv2.getOptimalDFTSize(cols) * pad_factor
sampling_period = pix * np.sin(theta) # microns
frequencies = np.fft.fftfreq(FFTsize, sampling_period) # spatial frequency in 1/microns of the Talbot pattern
Talbot_lengths = 1 / frequencies

plt.ion()
fig, ax = plt.subplots()

lamb = 0.632
zT = lamb / (1-np.sqrt(1-lamb**2/gratinglen**2))
ax.axvline(zT, color='r')

lamb = 0.532
zT = lamb / (1-np.sqrt(1-lamb**2/gratinglen**2))
plt.axvline(zT, color='g')

ax.set_xlim(0, 12)
line, = ax.plot(Talbot_lengths,np.zeros(Talbot_lengths.shape))

while True:
    # Acquisition
    ret, frame = cam.read()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("talbot", gray)
    #print(gray.shape)
    
    #k = cv2.waitKey(1000)
    
    # Filtering
    filtered = filters.difference_of_gaussians(gray, 0, 10)
    #cv2.imshow("talbot", filtered)
    
    #k = cv2.waitKey(1000)
    
    # FFT
    FFT = np.zeros(FFTsize)
    for i in range(filtered.shape[0]):
        sig = np.concatenate((filtered[i, :], np.zeros(FFTsize - filtered.shape[1])))
        fft = np.fft.fft(sig)
        FFT = FFT + fft
    out = np.abs(FFT)
    
    # rows, cols = filtered.shape
    # m = cv2.getOptimalDFTSize(rows) * pad_factor
    # n = cv2.getOptimalDFTSize(cols) * pad_factor
    # padded = cv2.copyMakeBorder(filtered, 0, m - rows, 0, n - cols, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    # planes = [np.float32(padded), np.zeros(padded.shape, np.float32)]
    # complexpadded = cv2.merge(planes)
    # cv2.dft(complexpadded, complexpadded, cv2.DFT_ROWS)
    # cv2.split(complexpadded, planes) # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    # cv2.magnitude(planes[0], planes[1], planes[0])# planes[0] = magnitude
    # magnitude = planes[0]
    #cv2.imshow("talbot", magnitude)
    #out = np.sum(magnitude, axis=0)
    #print(out.shape)
    
    line.set_ydata(out)
    ax.set_ylim(0, 2500)
    fig.canvas.draw()
    fig.canvas.flush_events()

    k = cv2.waitKey(1000)
    if k%256 == 27:
        # ESC pressed
        print("Esc")
        break

cam.release()

cv2.destroyAllWindows()