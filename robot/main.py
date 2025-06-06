#from camera.main_camera import is_colision
#import RPi.GPIO as GPIO
"""from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306, sh1106"""
import time
from PIL import ImageFont
#from Raspblock import Raspblock
import numpy as np
import string
#import serial


class Vendangeuse():

    def __init__(self):
        GPIO.setmode(GPIO.BCM) 
        #PIN
        self.MAGNET=19
        self.START=13
        self.PAMI=6
        self.COL = 5

        # robot
        self.ROBOT=Raspblock()
        self.WHEEL_RADIUS = 0.0325  # Rayon des roues en mètres
        self.WHEEL_BASE = 0.21 # Distance entre les roues en mètres
        self.WHEEL_CIR= 2*np.pi*self.WHEEL_RADIUS
        self.ser = serial.Serial("/dev/ttyAMA0", 115200)
        self.ser.flushInput()
        
        #VAR 
        self.orientation_h = None
        self.SPEED=2



        self.setup()

    def setup(self):
        """
        fonction qui setup le robot au démarage
        """
        GPIO.setup(self.MAGNET, GPIO.OUT)
        GPIO.setup(self.START, GPIO.IN)
        self.control_magnet(0) # on desactive l'aimant
        self.show_score(0) # on affiche un score de 0 
        self.control_pami(0)
        while self.orientation_h==None:
            self.orientation_h=self.get_orientation()
            time.sleep(0.1)
        


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
            recv = list(self.ser.read(count))
            recv = str(bytes(recv), encoding='UTF-8')
            if( recv.find("{") != -1 and recv.find("}#") != -1 ):
                #print(recv)
                #reg = re.compile('^{A(?P<Pitch>[^ ]*):(?P<Roll>[^ ]*):(?P<Yaw>[^ ]*):(?P<Voltage>[^ ]*)}#')
                return recv
        self.ser.flushInput()

    def get_board_info(self, i):
        self.ROBOT.BoardData_Get(i)  # Get voltage data
        data=self.Attitude_update()
        if type(data) ==str:
            return data.strip('{|}#').split(':')[-1]
        else:
            return None

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
    
    def control_pami(self,activate):
        if activate==0:
            GPIO.output(self.PAMI,GPIO.Low)
        elif activate==1:
            GPIO.output(self.PAMI, GPIO.HIGH)
        else:
            print(f"Warning: code not valid, must be 1 or 0 not {activate}")
    
    def proximity(self):
        return not GPIO.input(self.COL)

    def get_orientation(self):
        ori=self.get_board_info(8)
        if ori == 'None':
            return None
        else:
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
        start = 1
        while start:
            start = GPIO.input(self.START)
        return time.time()

    def move(self, HG,BG, HD,BD):
        self.ROBOT.Speed_Wheel_control(HD,BD,BG,HG)
    
    def choose_move(self, distance,orientation):
        # 180-0
        orientation=np.rad2deg(orientation)
        new_orientation =orientation + self.orientation_h
        if new_orientation-(90+self.orientation_h) < 0:
            Rcoef=-1
            Lcoef=1
        else:
            Rcoef=1
            Lcoef=-1


        while (self.get_orientation() < new_orientation+5) and ((self.get_orientation() > new_orientation+5)):
            self.move( Lcoef*self.SPEED, Lcoef*self.SPEED, Rcoef*self.SPEED, Rcoef*self.SPEED)

    def close_connection(self):
        del self.ROBOT




robot = Vendangeuse()
V=2

start = robot.on_start()
print("start")
while True:
    print("recule")
    for i in range(5000):
        while robot.proximity():
            pass
        robot.move(V,V,V,V)
        time.sleep(0.001)
    print("avance et depose la banière")
    for i in range(5000):
        while robot.proximity():
            pass
        robot.move(-V,-V,-V,-V)
        time.sleep(0.001)
    print("attend", time.time()-start)
    while time.time()-start<85:
        print("attend", time.time()-start)
        time.sleep(1)
    print("start pami")
    robot.contol_pami(1)
    break
    

