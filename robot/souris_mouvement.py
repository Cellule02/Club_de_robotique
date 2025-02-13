from pynput import*

def obtenir_coords(x, y):
    return(x, y)

def souris_ecoute():
    with mouse.Listener(on_move = obtenir_coords) as listen:
        listen.join()




