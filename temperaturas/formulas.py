import random
import pandas as pd

def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius):
    return celsius + 273.15

def kelvin_to_celsius(kelvin):
    return kelvin - 273.15

#Create a csv file with 100 random temperatures in Celsius and the equivalent temperatures in Fahrenheit
#Random float of temperatures between -100 and 100
temperatures_celsius = [round(random.uniform(-100, 100),3) for i in range(100)]
temperatures_fahrenheit = [round(celsius_to_fahrenheit(temp),3) for temp in temperatures_celsius]

df = pd.DataFrame({'Celsius': temperatures_celsius, 'Fahrenheit': temperatures_fahrenheit})
df.to_csv('temperaturas/temperatures.csv', index=False)