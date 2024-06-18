import sys
import matplotlib

matplotlib.use('Qt5Agg')
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QComboBox, QCheckBox, QSizePolicy,QLineEdit
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torch
import torch.nn as nn
#
from PyQt5.QtWidgets import QApplication,QLineEdit,QWidget,QFormLayout
from PyQt5.QtGui import QIntValidator,QDoubleValidator,QFont
from PyQt5.QtCore import Qt

import matplotlib.pyplot as plt
import dirac_gan


class Canvas(FigureCanvas):
    """
    General canvas class
    """

    def __init__(self, parent: "App" = None, width: int = 5, height: int = 4, dpi: int = 100) -> None:
        """
        Constructor method
        :param parent: (App) Parent app
        :param width: (int) Width of the plot
        :param height: (int) Height of the plot
        :param dpi: (int) DPI to be utilized
        """
        # Init figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        # Call super constructor and set parent
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        # Set some properties
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)


class DynamicCanvas(Canvas):
    """
    A dynamic canvas.
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Constructor method
        :param args: Arguments
        :param kwargs: Key word arguments
        """
        # Call super constructor
        Canvas.__init__(self, *args, **kwargs)
        # Initialize figure
        self.compute_initial_figure()

    def compute_initial_figure(self) -> None:
        """
        Method sets the initial figure
        """
        self.axes.set_xlim((-1.1, 1.1))
        self.axes.set_ylim((-2.1, 2.1))
        self.axes.grid()

    def update_figure(self, parameters: torch.Tensor, gradients: torch.Tensor, parameter_history: torch.Tensor,*param) -> None:
        """
        Method updates figure
        :param parameters: (torch.Tensor) Parameters of vector field [n, 2]
        :param gradients: (torch.Tensor) Gradients vectors of vector field [n, 2]
        :param parameter_history: (torch.Tensor) Parameter history of training [m, 2]
        """
        self.axes.cla()
        self.axes.grid()
        self.axes.set_xlim((-2.1, 2.1))
        self.axes.set_ylim((-2.1, 2.1))
        self.axes.quiver(parameters[..., 0], parameters[..., 1], -gradients[..., 0], -gradients[..., 1])
        self.axes.scatter(parameter_history[..., 0], parameter_history[..., 1])
        self.axes.scatter([1.], [1.], color="red")
        self.axes.set_xlabel("$\\theta$")
        self.axes.set_ylabel("$\\psi$")
        self.draw()
        self.fig.savefig('images/%s_Reg_%s_IN_%s.pdf'%(param),dpi=100)
        torch.save(parameter_history,'param_history/%s_Reg_%s_IN_%s.pt'%(param))
        


class App(QMainWindow):
    """
    GUI application class
    """

    def __init__(self) -> None:
        """
        Constructor method
        """
        # Call super constructor
        super(App, self).__init__()
        # Init some attributes regarding the GUI
        self.left: int = 50
        self.top: int = 50
        self.title: str = "DiracGAN"
        self.width: int = 1000
        self.height: int = 800
        # General setting
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        # Make dynamic plot
        self.dynamic_plot = DynamicCanvas(self, width=7, height=7, dpi=100)
        ##Make parameter box
        #self.niter_list_box = QComboBox(self)
        #self.niter_list_box.addItem("32")
        #self.niter_list_box.addItem("64")
        #self.niter_list_box.addItem("128")
        #self.niter_list_box.resize(300, 50)
        #self.niter_list_box.move(self.width - 300, self.height - 350)
        ## Make parameter box
        #self.batch_list_box = QComboBox(self)
        #self.batch_list_box.addItem("32")
        #self.batch_list_box.addItem("64")
        #self.batch_list_box.addItem("128")
        #self.batch_list_box.resize(300, 50)
        #self.batch_list_box.move(self.width - 300, self.height - 300)
        ##Learning rate box
        #self.lr_list_box = QComboBox(self)
        #self.lr_list_box.addItem("32")
        #self.lr_list_box.addItem("64")
        #self.lr_list_box.addItem("128")
        #self.lr_list_box.resize(300, 50)
        #self.lr_list_box.move(self.width - 300, self.height - 300)
        # Make list box to choose regularization
        self.instance_noise_box = QCheckBox("Instance noise", self)
        self.instance_noise_box.resize(300, 50)
        self.instance_noise_box.move(self.width - 300, self.height - 200)
        # Make list box to choose regularization
        self.regularization_box = QComboBox(self)
        self.regularization_box.addItem("None")
        self.regularization_box.addItem("R1 gradient penalty")
        self.regularization_box.addItem("RLC")
        self.regularization_box.addItem("R2 gradient penalty")
        self.regularization_box.addItem("CLC")
        self.regularization_box.resize(300, 50)
        self.regularization_box.move(self.width - 300, self.height - 150)
        # Make list box to choose GAN loss
        self.gan_list_box = QComboBox(self)
        self.gan_list_box.addItem("Standard GAN")
        self.gan_list_box.addItem("Non-saturating GAN")
        self.gan_list_box.addItem("Wasserstein GAN")
        self.gan_list_box.addItem("Wasserstein GAN GP")
        self.gan_list_box.addItem("Least squares GAN")
        self.gan_list_box.addItem("Hinge GAN")
        self.gan_list_box.addItem("MCGAN")
        self.gan_list_box.resize(300, 50)
        self.gan_list_box.move(self.width - 300, self.height - 100)
        # Make run button
        run_button = QPushButton("button", self)
        run_button.setText("Run training")
        run_button.resize(300, 50)
        run_button.move(self.width - 300, self.height - 50)
        run_button.clicked.connect(self.__train)
        # Show everything
        self.show()

    def __train(self) -> None:
        """
        Method performs DiracGAN training and plots the training process
        """
        # Check if instance noise is utilized
        instance_noise: bool = self.instance_noise_box.isChecked()
        # Check which regularization is utilized
        regularization: str = self.regularization_box.currentText()
        # Check which GAN loss is utilized
        gan_loss: str = self.gan_list_box.currentText()
        # Init generator and discriminator
        generator: nn.Module = dirac_gan.Generator()
        discriminator: nn.Module = dirac_gan.Discriminator()
        # Init loss functions
        MC=False
        if gan_loss == "Standard GAN":
            generator_loss: nn.Module = dirac_gan.GANLossGenerator()
            discriminator_loss: nn.Module = dirac_gan.GANLossDiscriminator()
        elif gan_loss == "Non-saturating GAN":
            generator_loss: nn.Module = dirac_gan.NSGANLossGenerator()
            discriminator_loss: nn.Module = dirac_gan.NSGANLossDiscriminator()
        elif gan_loss == "Wasserstein GAN":
            generator_loss: nn.Module = dirac_gan.WassersteinGANLossGenerator()
            discriminator_loss: nn.Module = dirac_gan.WassersteinGANLossDiscriminator()
        elif gan_loss == "Wasserstein GAN GP":
            generator_loss: nn.Module = dirac_gan.WassersteinGANLossGPGenerator()
            discriminator_loss: nn.Module = dirac_gan.WassersteinGANLossGPDiscriminator()
        elif gan_loss == "Least squares GAN":
            generator_loss: nn.Module = dirac_gan.LSGANLossGenerator()
            discriminator_loss: nn.Module = dirac_gan.LSGANLossDiscriminator()
        elif gan_loss=="MCGAN":
            generator_loss: nn.Module = dirac_gan.MCGANLossGenerator()
            discriminator_loss: nn.Module = dirac_gan.MCGANLossDiscriminator()
            MC=True
        else:
            generator_loss: nn.Module = dirac_gan.HingeGANLossGenerator()
            discriminator_loss: nn.Module = dirac_gan.HingeGANLossDiscriminator()
        # Init regularization loss
        if regularization == "None":
            regularization_loss: nn.Module = None
        elif regularization == "R1 gradient penalty":
            regularization_loss: nn.Module = dirac_gan.R1()
        elif regularization == "RLC":
            regularization_loss: nn.Module = dirac_gan.RLC()
        elif regularization == "CLC":
            regularization_loss: nn.Module = dirac_gan.CLC()
        else:
            regularization_loss: nn.Module = dirac_gan.R2()

        # Init optimizers
        generator_optimizer: torch.optim.Optimizer = torch.optim.SGD(params=generator.parameters(),
                                                                     lr=dirac_gan.HYPERPARAMETERS["lr"],
                                                                     momentum=0.)
        discriminator_optimizer: torch.optim.Optimizer = torch.optim.SGD(params=discriminator.parameters(),
                                                                         lr=dirac_gan.HYPERPARAMETERS["lr"],
                                                                         momentum=0.)
        # Make model wrapper
        model_wrapper = dirac_gan.ModelWrapper(generator=generator,
                                               discriminator=discriminator,
                                               generator_optimizer=generator_optimizer,
                                               discriminator_optimizer=discriminator_optimizer,
                                               generator_loss_function=generator_loss,
                                               discriminator_loss_function=discriminator_loss,
                                               regularization_loss=regularization_loss,MC=MC)
        # Get trajectory
        parameters, gradients = model_wrapper.generate_trajectory(instance_noise=instance_noise)
        # Perform training
        parameter_history: torch.Tensor = model_wrapper.train(instance_noise=instance_noise)
        # Plot results
        print(parameter_history[-10:, 0].mean().item(), parameter_history[-10:, 1].mean().item())
        self.dynamic_plot.update_figure(parameters, gradients,parameter_history,gan_loss,regularization,instance_noise)
        


if __name__ == '__main__':
    application = QApplication(sys.argv)
    _ = App()
    sys.exit(application.exec_())
