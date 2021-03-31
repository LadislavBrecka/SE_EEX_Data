import requests
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from sklearn.svm import SVR
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

class Handler:
    def __init__(self):
        # plt.ion()
        self.url = "https://seasweb.azurewebsites.net/data.json"
        self.filter_const = 25
        self.by1 = None
        self.by2 = None
        self.py1 = None
        self.py2 = None
        self.unfiltered_series = None
        self.filtered_series = None
        self.data_to_plot = None
        self.subplot_axes= None
        self.radio = None
        self.slider = None
        self.func = None
        self.df = None

    # Downloading data and creating dataset
    def create_dataset(self):
        # Loading dataset from given url
        rawData = json.loads(requests.get(self.url).text)
        self.df = pd.DataFrame(rawData)
        # Creating numpy arrays for further processing
        self.by1 = self.df['BY1'].astype('float').to_numpy()
        self.by2 = self.df['BY2'].astype('float').to_numpy()
        self.py1 = self.df['PY1'].astype('float').to_numpy()
        self.py2 = self.df['PY2'].astype('float').to_numpy()
        # Creating lists of series
        self.unfiltered_series = [self.by1, self.by2, self.py1, self.py2]
        self.filtered_series = [self.__filter(self.by1), self.__filter(self.by2), self.__filter(self.py1), self.__filter(self.py2)]
    
    # Plotting all series together on one figure
    def plot_all(self):
        # Testing if dataset was created
        if self.df is None:
            raise ValueError("Dataset is not created")
        
        # Plotting and setting up graph
        ax1 = self.df.plot(x='category', figsize=(10,9))
        ax1.title.set_text('European Energy Exchange (EEX) data for years 2021 and 2022')
        ax1.set_ylabel('Price[€]', fontsize='large', fontweight='bold')
        ax1.set_xlabel('Date', fontsize='large', fontweight='bold')
        ax1.legend(["BY1 BaseLoad 2021 in €/MWh", "BY2 BaseLoad 2022 in €/MWh", "PY1 PeakLoad 2021 in €/MWh", "PY2 PeakLoad 2022 in €/MWh", "CO2 - Price of emission allowances in €/tonne"])
        plt.draw()

    # Plotting all series separately on subplots
    def plot_by_one(self):
        # Testing if dataset was created
        if self.df is None:
            raise ValueError("Dataset is not created")

        # Plotting and setting up graph
        fig2, self.subplot_axes = plt.subplots(2, 2, figsize=(10,9))
        fig2.subplots_adjust(left=0.3, wspace=0.2, hspace=0.3)
        fig2.suptitle('All series separately')
        fig2.text(0.25, 0.5, 'Price[€/MWh]', rotation='vertical', verticalalignment='center', fontsize='large', fontweight='bold')
        fig2.text(0.6, 0.03, 'Date', fontsize='large', horizontalalignment='center', fontweight='bold')
        fig2.text(0.05, 0.1, 'Use slider only when filtered series is selected\nSlider for changing filter constant (filtering rate)', rotation='vertical',fontsize='large', fontweight='bold')
        
        # Reshaping list of axes for usage in for loop
        self.subplot_axes = np.reshape(self.subplot_axes, 4, 'F')
        # Updating which data will be plotted
        self.data_to_plot = self.unfiltered_series
        # Updating subplots by class function
        self.__subplot_update()

        # Creating radio button for changing which series to plot
        rax = plt.axes([0.05, 0.7, 0.15, 0.15])
        self.radio = RadioButtons(rax, ('Unfiltered series', 'Filtered series'))
        self.func = {'Unfiltered series': self.unfiltered_series, 'Filtered series': self.filtered_series}

        # Creating slider for changing filter constant
        axSlider = plt.axes([0.1, 0.1, 0.05, 0.5])
        self.slider = Slider(axSlider, 'Slider', valmin=1, valmax=125, valinit=25, orientation='vertical', valfmt='%d')

        # Assign a function handler to a button and slider
        self.radio.on_clicked(self.__radioButton_update)
        self.slider.on_changed(self.__slider_update)
        plt.draw()

    # Printing max values
    def print_max_values(self):
        # Printing 5 highest values to console for every series
        print("--------------------------------------------------")
        print("Highest values of series [BY1, BY2, PY1, PY2] :\n")
        # Iterating throught all series
        for s in self.unfiltered_series:
            # Indirect partition
            ind = np.argpartition(-s, 5)[:5]
            a = s[ind]
            # Swapping order
            a = -np.sort(-a)
            print(a)
        print("--------------------------------------------------")

    # Regression analysis
    def nonlinear_regression(self):
        # Prepocessing data for reggresion
        fig3 = plt.figure(3, figsize=(10,9))
        X = [i for i in range(len(self.unfiltered_series[0]))]
        X = np.asfarray(X).reshape(-1, 1)
        y = self.unfiltered_series[0]
        
        # Fit regression model
        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

        # Plot graph of reggresion
        plt.plot(X, svr_rbf.fit(X, y).predict(X), color='m', lw=2,
                  label='{} model'.format('RBF'))

        # Plot points
        plt.scatter(X,y, s=10)

        # Displaying information
        a = svr_rbf.score(X, y)
        fig3.suptitle('RBF regression')
        fig3.text(0.5, 0.9, 'Coefficient of determination R^2 is %s'%(a), horizontalalignment='center', fontsize='large', fontweight='bold')
        plt.ylabel('Price[€]', fontsize='large', fontweight='bold')
        plt.xlabel('Days', fontsize='large', fontweight='bold')
        plt.draw()

    # The function which handle subplots updating every time data to plot or slider value was changed
    def __subplot_update(self):
        self.__plot_on_axis()

        # Positioning x label elements
        for a in self.subplot_axes:
            plt.setp(a.get_xticklabels(), rotation=30, ha='right')

        # Naming each subplot
        self.subplot_axes[0].title.set_text('BY1')
        self.subplot_axes[1].title.set_text('BY2')
        self.subplot_axes[2].title.set_text('PY1')
        self.subplot_axes[3].title.set_text('PY2')

        plt.draw()

    # Handler for radio button
    def __radioButton_update(self, label):
       # Updating which data will be plotted
        self.data_to_plot = self.func[label]
        # Updating subplots by class function
        self.__subplot_update()
        # Reseting slider
        self.slider.reset()

    # Handler for slider
    def __slider_update(self, val):
        # Slider is working only in filtered series state
        if self.radio.value_selected == 'Filtered series':
            # Updating filter constant
            self.filter_const = int(self.slider.val)
            # Updating list of filtered series
            self.filtered_series = [self.__filter(self.by1), self.__filter(self.by2), self.__filter(self.py1), self.__filter(self.py2)]
            # Updating which data to plot
            self.data_to_plot = self.filtered_series
            # Updating subplots by class function
            self.__subplot_update()

    # The function for plotting columns of dataset to separated subplots
    def __plot_on_axis(self):
        # Each axis, one graph
        if len(self.data_to_plot) != len(self.subplot_axes):
            raise ValueError('This function plot one column of dataset stored in array on one axis. Data array length is not the same as axes array length.')
        i = 0
        for a in self.subplot_axes:
            a.clear()
            a.plot(pd.to_datetime(self.df['category']), self.data_to_plot[i])
            i += 1

    # The filter, it uses Furier transformation
    # ???!Malokedy vyuzijem v praxi nieco co som sa naucil v skole, ale toto je jedna z tych veci ktore som pochopil a pouzil!???
    def __filter(self, input):
        furrier_transform = np.fft.fft(input)
        shifted_furrier_transform = np.fft.fftshift(furrier_transform)
        HP_filter = np.zeros(len(shifted_furrier_transform), dtype=int)
        n = int(len(HP_filter))
        HP_filter[int(n/2) - self.filter_const : int(n/2) + self.filter_const] = 1
        output = shifted_furrier_transform * HP_filter
        output = abs(np.fft.ifft(output))

        return output      
pass


handler = Handler()
handler.create_dataset()
handler.plot_all()
handler.plot_by_one()
handler.nonlinear_regression()
handler.print_max_values()
plt.show()