from camera.main_camera import is_colision
import RPi.GPIO as GPIO
import time
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306, sh1106
import time
from PIL import ImageFont
from Raspblock import Raspblock
import numpy as np
import string
import serial


class Vendangeuse():

    def __init__(self):
        GPIO.setmode(GPIO.BCM) 
        #PIN
        self.MAGNET=19
        self.START=13

        # robot
        self.ROBOT=Raspblock()
        self.WHEEL_RADIUS = 0.0325  # Rayon des roues en mètres
        self.WHEEL_BASE = 0.21 # Distance entre les roues en mètres
        self.WHEEL_CIR= 2*np.pi*self.WHEEL_RADIUS

        self.ser = serial.Serial("/dev/ttyAMA0", 115200)
        self.ser.flushInput()

        #VAR 
        self.orientation_h = 0



        self.setup()

    def setup(self):
        """
        fonction qui setup le robot au démarage
        """
        GPIO.setup(self.MAGNET, GPIO.OUT)
        GPIO.setup(self.START, GPIO.IN)
        self.control_magnet(0) # on desactive l'aimant
        self.show_score(0) # on affiche un score de 0 
        self.orientation_h=self.get_orientation(8)


    def cam_data(self):
        import socket
        socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        ip = "0.0.0.0"
        port = 5005
        serverAddress = (ip, port)
        socket.bind(serverAddress)
        socket.listen(1)
        print("Waiting for connection")
        connection, add = socket.accept()
        data = connection.recv(2048)
        return data

    def Attitude_update(self):
        # Get receive buffer character
        count = self.ser.inWaiting()
        if count != 0:
            recv = list(self.read(count))
            recv = str(bytes(recv), encoding='UTF-8')
            if( recv.find("{") != -1 and recv.find("}#") != -1 ):
                #print(recv)
                #reg = re.compile('^{A(?P<Pitch>[^ ]*):(?P<Roll>[^ ]*):(?P<Yaw>[^ ]*):(?P<Voltage>[^ ]*)}#')
                return recv
        self.ser.flushInput()

    def get_board_info(self, i):
        self.ROBOT.BoardData_Get(i)  # Get voltage data
        data=self.Attitude_update()
        return data[i]

    def control_magnet(self,activate=0):
        """
        @param:
            -activate :bool: 1 si on active l'aimant, 0 pour le désactiver
        @return:
            -None
        """
        if activate==0:
            GPIO.output(self.MAGNET,GPIO.HIGH)
        elif activate==1:
            GPIO.output(self.MAGNET, GPIO.LOW)
        else:
            print(f"Warning: code not valid, must be 1 or 0 not {activate}")

    def get_orientation(self):
        return self.get_board_info(8)/100
    
    def show_score(self, score):
        # Initialize the I2C interface
        serial = i2c(port=1, address=0x3C)

        # Initialize the device (choose one based on your display)
        # For SSD1306-based display:
        device = ssd1306(serial, width=128, height=32)
        # For SH1106-based display (uncomment if needed):
        # device = sh1106(serial, width=128, height=64)

        large_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)

        # Display some text
        with canvas(device) as draw:
            draw.text((0, 0), f"score: {score}", fill="white", font=large_font)
    
    def on_start(self):
        """
        fonction qui lance le jeux de 100sec
        """
        start = 0
        while not start:
            start = GPIO.input(self.START)
        return start

    def move(self, HG,BG, HD,BD):
        self.ROBOT.Speed_Wheel_control(HD,BD,BG,HG)
    
    def choose_move(self, distance,orientation):
        # 180-0
        orientation = self.get_orientation() - self.orientation_h


    def close_connection(self):
        del self.ROBOT





        