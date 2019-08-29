import numpy as np
import pandas as pd
import cv2, os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Image, PageBreak, Table, TableStyle
from reportlab.lib.units import cm
from reportlab.lib import utils
from reportlab.lib import colors
from tqdm import tqdm
import matplotlib.gridspec as gridspec

"""
    The ``validation`` module
    ======================
 
    Permet de créer des rapports en chargeant un modèle et ses poids sur un patch de validation.
 
    :Example:
 
    >>> from validation import Validation
    >>> valid = Validation('/home/alex/output.csv', '/home/alex/images/')
    >>> valid.report('/home/alex/report.pdf')
"""





# Liste des classes, peut être changée sans problème
patho2int, int2patho = {}, {}
    
def list_to_dict(L):
    class2int, int2class = {}, {}
    for i, name in enumerate(L):
        class2int[name] = i
        int2class[i] = name
    return class2int, int2class

class Validation:
    def __init__(self, 
                 df,
                 image_path  = '',
                 x_name      = 'file',
                 y_true_name = 'y_true'
                 ):
        
        global patho2int
        global int2patho
        
        if type(df) == str:
            df = pd.read_csv(df)
            if 'Unnamed: 0' in df.keys():
                df = df.drop('Unnamed: 0', axis = 1)
        
        if not image_path.endswith('/'):
            image_path = image_path + '/'
        
        self.image_path = image_path
        self.x          = df[x_name].tolist()
        self.y_true     = np.array(df[y_true_name].tolist())
        names           = df.keys().tolist()
        names.remove('y_true')
        names.remove('file')
        patho2int, int2patho = list_to_dict(names)
        self.y_prob          = np.zeros((len(self.y_true), len(names)))
        for i, name in enumerate(names):
            self.y_prob[:, i] = df[name]
        self.y_pred = np.argmax(self.y_prob, axis = 1)
        
        
        
    def confusion_matrix(self, show = True, normalize = False):
            
        self.matrix = confusion_matrix(self.y_true,
                                       self.y_pred,
                                       labels = [i for i in int2patho])
        
        if normalize:
            self.matrix = self.matrix.astype('float')
            sum_matrix  = self.matrix.sum(axis=1)[:, np.newaxis]
            sum_matrix[sum_matrix == 0] = 1000
            self.matrix = self.matrix / sum_matrix
        
        if show:
            return self.plot_confusion_matrix(normalize = normalize)
        
        return self.matrix
    
    
    def plot_confusion_matrix(self, normalize = False):  
        cmap    = plt.cm.Blues
        classes = [i for i in patho2int]
        title   = 'Confusion matrix'
        fig, ax = plt.subplots(figsize = (11, 9))
        im = ax.imshow(self.matrix, interpolation='nearest', cmap=cmap)
        
        ax.figure.colorbar(im, ax=ax)
        
        ax.set(xticks      = np.arange(self.matrix.shape[1]),
               yticks      = np.arange(self.matrix.shape[0]),
               xticklabels = classes, 
               yticklabels = classes,
               title       = title,
               ylabel      = 'True label',
               xlabel      = 'Predicted label')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        fmt = '.2f' if normalize else 'd'
        thresh = self.matrix.max() / 2.
        for i in range(self.matrix.shape[0]):
            for j in range(self.matrix.shape[1]):
                ax.text(j, i, format(self.matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if self.matrix[i, j] > thresh else "black")

        ax.set_xticks(np.arange(0, len(patho2int), 0.5), minor=True)
        ax.set_yticks(np.arange(0, len(patho2int), 0.5), minor=True)
        ax.grid(which='minor')
        
        ax.set_ylim(bottom = self.matrix.shape[0] - 0.5, top = -0.5)
        fig.tight_layout()
        return fig

    
    def random_patch_plot(self, info = False):
        self.patch_plot(np.random.randint(self.n), info)
    
    def load(self, x):
        if type(x) == str:
            return cv2.imread(self.image_path + x)
        if type(x) == list:
            return np.array([cv2.imread(i) for i in x])
    
    
    def patch_plot(self, i, info = False):
        fig    = plt.figure(figsize=(9, 9))
        ax     = fig.add_subplot(111)
        data   = self.load(self.x[i])
        classe = self.y_true[i]
        proba  = self.y_prob[i]
        patch  = Patch(data, classe, proba)
        patch.plot(ax, info)
        
       
    def multiple_patch_plot(self, display = None, info = False, color = np.array([])):
        n       = len(display)
        column  = 3
        row     = (n // 3) + 1
        fig, ax = plt.subplots(nrows=row, ncols=column, figsize = (18, 21))
        for i, axi in enumerate(ax.flat[:n]):
            no_image = display[i]
            data     = self.load(self.x[no_image])
            classe   = self.y_true[no_image]
            proba    = self.y_prob[no_image]
            patch    = Patch(data, classe, proba)
            patch.multiple_plot(axi, info)
        for axi in ax.flat[n:]:
            fig.delaxes(axi)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        if color.size:
            for axi, color in zip(ax.flat[:n], color):
                plt.setp(axi.spines.values(), color=color)
        
        fig.show()
        
        
    def mini_patch_plot(self, display = None, color = np.array([])):
        n       = len(display)
        column  = 6
        row     = (n // 6) + 1
        fig, ax = plt.subplots(nrows=row, ncols=column, figsize = (18, (11/3)*row))
        for i, axi in enumerate(ax.flat[:n]):
            no_image = display[i]
            data     = self.data_x[no_image]
            classe   = self.y_true[no_image]
            proba    = self.y_prob[no_image]
            patch    = Patch(data, classe, proba)
            patch.multiple_plot(axi, True, True)
        for axi in ax.flat[n:]:
            fig.delaxes(axi)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        if color.size:
            for axi, color in zip(ax.flat[:n], color):
                plt.setp(axi.spines.values(), color=color)
                
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        fig.show() 
        return fig
    
        

    def plot_classe(self, classe, info = False, mini = False):
        patch_to_display = np.where(self.y_true == classe)[0]
        if not patch_to_display.any():
            return None
        color = np.array([])
        if info:
            color = np.array(['r' for _ in range(len(patch_to_display))])
            color[np.where(self.y_true[patch_to_display] == self.y_pred[patch_to_display])[0]] = 'g'
        if not mini:
            self.multiple_patch_plot(patch_to_display, info, color)
        else:
            return self.mini_patch_plot(patch_to_display, color)
            
    
    def plot_pred_classe(self, classe, info = False, mini = False):
        patch_to_display = np.where(self.y_pred == classe)[0]
        if not patch_to_display.any():
            return None
        color = np.array([])
        if info:
            color = np.array(['r' for _ in range(len(patch_to_display))])
            color[np.where(self.y_true[patch_to_display] == self.y_pred[patch_to_display])] = 'g'
        if not mini:
            self.multiple_patch_plot(patch_to_display, info, color)
        else:
            return self.mini_patch_plot(patch_to_display, color)
       
    
    def fake_positive(self, classe, info = False, mini = False):
        patch_to_display = np.where(self.y_pred != self.y_true)[0]
        patch_to_display = patch_to_display[self.y_pred[patch_to_display] == classe]
        if not patch_to_display.any():
            return None
        color = np.array(['r' for i in range(len(patch_to_display))]) if info else np.array([])
        if not mini:
            self.multiple_patch_plot(patch_to_display, info, color)
        else:
            return self.mini_patch_plot(patch_to_display, color)
    
    
    def true_positive(self, classe, info = False, mini = False):
        patch_to_display = np.where(self.y_pred == self.y_true)[0]
        patch_to_display = patch_to_display[self.y_pred[patch_to_display] == classe]
        if not patch_to_display.any():
            return None
        color = np.array(['g' for i in range(len(patch_to_display))]) if info else np.array([])
        if not mini:
            self.multiple_patch_plot(patch_to_display, info, color)
        else:
            return self.mini_patch_plot(patch_to_display, color)
    
    
    def fake_negative(self, classe, info = False, mini = False):
        patch_to_display = np.where(self.y_pred != self.y_true)[0]
        patch_to_display = patch_to_display[self.y_true[patch_to_display] == classe]
        if not patch_to_display.any():
            return None
        color = np.array(['r' for i in range(len(patch_to_display))]) if info else np.array([])
        if not mini:
            self.multiple_patch_plot(patch_to_display, info, color)
        else:
            return self.mini_patch_plot(patch_to_display, color)
        
        
    def info(self):
        print(classification_report(self.y_true, self.y_pred))
        
        
    def batch_info(self):
        patho       = [i for i in patho2int]
        fig, ax     = plt.subplots(figsize = (11, 9))
        data_true   = self.y_true
        data_pred   = self.y_pred
        data_sucess = [x for y, x in zip(data_true, data_pred) if y == x]
        counts, bins, patches = ax.hist([data_true, data_pred, data_sucess],
                                        bins        = range(len(patho2int) + 1), 
                                        orientation = 'horizontal',
                                        edgecolor   = 'gray',
                                        label       = ['Real', 'Predicted', 'Succeed'])
        
        ax.set(yticks = np.arange(0.5, len(patho2int) + 1.5), yticklabels = patho)
        #for i, v in enumerate(counts):
        #    ax.text(v - .005, i + .4, '{}'.format(v))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.legend()
        return fig, [data_true, data_pred, data_sucess]
    
    
    def class_info(self, patho, L):
        i = patho2int[patho]
        fig, ax     = plt.subplots(figsize = (3, 5))
        counts, bins, patches = ax.hist(L,
                                        bins        = range(len(patho2int)),
                                        edgecolor   = 'gray',
                                        label       = ['Real', 'Predicted', 'Succeed'])
        if len(counts[0]) > i:
            ax.set(xticks = np.linspace(i, i+1, 5)[1:-1], xticklabels = np.array(counts, dtype=np.int64)[:, i])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim(i, i+1)
        fig.legend()
        return fig
        
    
        
        
        
    def report(self, path, title = 'Rapport', fun = ['plot_classe', 'fake_negative']):
        page_size = 2304
        
        def get_image(path, width=1*cm):
            img = utils.ImageReader(path)
            iw, ih = img.getSize()
            aspect = ih / float(iw)
            return Image(path, width=width, height=(width * aspect))

        my_doc             = SimpleDocTemplate(path)
        sample_style_sheet = getSampleStyleSheet()
        story              = []
        
        title = Paragraph("Report on a validation batch", sample_style_sheet['Heading1'])
        
        batch_info_fig, info_list = self.batch_info()
        batch_info_fig.savefig('temp_info_batch.png')
        plt.close(batch_info_fig)
        batch_image = get_image('temp_info_batch.png', 19*cm)

        fig = self.confusion_matrix(normalize = True)
        fig.savefig('temp_matrix.png')
        plt.close(fig)
        matrix_confusion = get_image('temp_matrix.png', 22*cm)

        patch_display = [[Paragraph(name, sample_style_sheet['Heading1'])] for name in fun]

        title_style = sample_style_sheet['Heading1']
        title_style.aligment = 1
        pages = {}
        for function in fun:
            for patho in patho2int:
                pages[function + patho] = Page('temp_{}_{}.png'.format(function, patho))
                
        for i, x in enumerate(tqdm(self.x)):
            patch = Patch(self.load(x), self.y_true[i], self.y_prob[i])
            color = 'green' if self.y_true[i] == self.y_pred[i] else 'red'
            fig   = plt.figure(figsize = (5, 8))
            fig   = patch.single_plot_hist(fig, color)
            fig.savefig('temp__.png')
            plt.close(fig)
            image = cv2.imread('temp__.png')
            if 'plot_classe' in fun:
                pages['plot_classe' + int2patho[self.y_true[i]]].add_image(image)
            if 'fake_negative' in fun and color == 'red':
                pages['fake_negative' + int2patho[self.y_true[i]]].add_image(image)
        for j, function in enumerate(fun): 
            for patho in patho2int:
                img = cv2.imread('temp_{}_{}.png'.format(function, patho))
                im = []
                split = [img[k*page_size:min((k+1)*page_size, img.shape[0]), ...] 
                            for k in range(1 + (img.shape[0] // page_size))]
                for k, splitted in enumerate(split):
                    if splitted.any():
                        cv2.imwrite('temp_{}_{}_{}.png'.format(function, patho, k), splitted)
                        im.append(get_image('temp_{}_{}_{}.png'.format(function, patho, k), 14.5*cm))  

                title_section = Paragraph('{}<br/>{}'.format(function, patho), title_style)
                fig = self.class_info(patho, info_list)
                fig.savefig('temp_page_{}_{}.png'.format(function, patho))
                plt.close(fig)
                class_image = get_image('temp_page_{}_{}.png'.format(function, patho), 4*cm)
                patch_display[j].append(([title_section, class_image], im))
                    
                
        
        story.append(title)
        story.append(batch_image)
        story.append(matrix_confusion)
        story.append(PageBreak())
        
        
        for data in patch_display:
            #story.append(data[0])
            for tupl in data[1:]:
                story.append(tupl[0][0])
                story.append(tupl[0][1])
                for img in tupl[1]:
                    story.append(img)
                    story.append(PageBreak())
        
        my_doc.build(story)
        file_to_remove = os.listdir()
        file_to_remove = [i for i in file_to_remove if i.startswith('temp_')]
        for i in file_to_remove:
            os.remove(i)
        print('done')
        
        
class Patch:
    def __init__(self, data, classe, proba):
        self.data   = data
        self.classe = classe
        self.proba  = proba
        self.pred   = np.argmax(proba)
        
    def plot(self, ax, info = False):
        if info:
            info = ''
            for i, prob in enumerate(self.proba):
                prob  = str(prob*100)[:4]
                patho = int2patho[i]
                info += '\n{} : {}%'.format(patho, prob)
            ax.text(440, 205, info, ha='right', fontsize=10)
            ax.text(6, 10, int2patho[self.classe])
        data = cv2.resize(self.data, (448, 448))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none') 
        ax.imshow(data.astype(np.uint8))
        
        
    def multiple_plot(self, ax, info = False, mini = False):
        ax.imshow(self.data.astype(np.uint8))
        if info and not mini:
            ax.text(6, 10, 'Real : ' + int2patho[self.classe], fontsize = 15)
            ax.text(6, 22, 'Predicted : ' + int2patho[self.pred], fontsize = 15)
        elif info:
            ax.set_title('Real : ' + int2patho[self.classe] + '\nPredicted : ' + int2patho[self.pred])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        [i[1].set_linewidth(2) for i in ax.spines.items()]
        
    def single_plot_hist(self, fig, color):
        data = cv2.resize(self.data, (448, 448))
        gs   = gridspec.GridSpec(6, 1, figure = fig)
        gs.update(wspace=0.025, hspace=0.05)
        
        # Partie où l'on va afficher l'image
        ax   = plt.subplot(gs[:4])
        ax.imshow(data.astype(np.uint8))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        [i[1].set_linewidth(2) for i in ax.spines.items()]
        plt.setp(ax.spines.values(), color=color)
        
        # Partie dédiée à l'histogramme
        ax      = plt.subplot(gs[4:5])
        # Tous les axes font la même taille
        ax.set_xlim([0, 1])
        maxprob = np.sort(self.proba)[-3:]
        index   = [int2patho[np.where(self.proba == i)[0][0]] for i in maxprob]
        counts, bins, patches  = ax.hist(range(3), 
                                         bins        = [-0.45, 0.45 , 0.6, 1.45, 1.6, 2.6], 
                                         orientation = 'horizontal', 
                                         weights     = maxprob,
                                         edgecolor   = 'gray')
        y = bins[:-1] + ((bins[1]+bins[0])/2)*np.ones(len(bins) -1)
        y_label = [''] * (len(index) * 2 - 1)
        y_label[0::2] = index
        ax.set(yticks      = y, 
               yticklabels = [])
        # Pour éviter d'écrire en blanc sur du blanc, lorsque la barre est trop petite on écrit en noir
        for i in [0, 2, 4]:
            color = 'white' if maxprob[i//2] > 0.35 else 'black'
            ax.text(0.02 * maxprob[-1], y[i] + 0.25 + 0.05*(i == 4), y_label[i], fontsize = 13, color = color)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('none')
        xticks = ax.xaxis.get_major_ticks()
        xticks[0].label1.set_visible(False)
        return fig
    
    
class Page:
    """
    L'objet page sert à rajouter des images au fur et à mesure sur une page en ajoutant des lignes autant que nécessaire
    """
    def __init__(self, path):
        self.row    = 0
        self.column = 0
        self.path   = path
        self.index  = 0
        cv2.imwrite(path, self.new_line())
        
    def new_line(self):
        return 255*np.ones((576, 360*4, 3))
    
    def add_image(self, image):
        img = cv2.imread(self.path)
        
        if self.column == 0 and self.index != 0:
            current_line = self.new_line()
            img = np.concatenate((img, current_line), axis=0)
        image = cv2.resize(image, (360, 576))
        img[576*self.row:576*(self.row + 1), self.column*360:(self.column + 1)*360, :] = image
        cv2.imwrite(self.path, img)
        
        self.index += 1
        self.column = self.index %  4
        self.row    = self.index // 4
        
if __name__ == '__main__':
    df_path      = input('Path to Pandas Dataframe\n')
    image_path   = input('Folder containing images\n')
    save_path    = input('Path to save the pdf\n')
    custom_name  = input('Do you want to use custom name for datafrma ? y/n\n')
    if custom_name == 'y':
        x_name = input('Name of the column containing file names')
        y_true = input('Name of the column containning y labels')
    else:
        x_name = 'file'
        y_true = 'y_true'
        
    validation = Validation(df_path,
                            image_path,
                            x_name = x_name,
                            y_true_name = y_true
                            )
    
    validation.report(save_path, 'Rapport', fun = ['plot_classe', 'fake_negative'])
    
        