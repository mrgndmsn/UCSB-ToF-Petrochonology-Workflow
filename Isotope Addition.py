#/ Type: DRS
#/ Name: ToF Isotope addition
#/ Authors: Morgan Adamson
#/ Description: Combined isotope ppm, modified from Joe Petrus and Paul Bence 3D Trace Elements. Only modified to add together cps from non interfered isotopes for each element.
#/ References: Petrus and Bence, 2023
#/ Version: 1.3
#/ Contact: mna@ucsb.edu

"""
Copyright (c) 2022 iolite software

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from iolite.QtGui import QLabel, QAbstractTableModel, QTableView, QTabWidget, QWidget, QSortFilterProxyModel, QLineEdit
from iolite.QtGui import QToolButton, QMenu, QWidgetAction, QCheckBox,QAbstractItemView, QColor, QComboBox, QDoubleSpinBox, QColorDialog
from iolite.QtGui import QWidget, QVBoxLayout, QSizePolicy, QDialog, QHBoxLayout, QShortcut, QKeySequence, QGroupBox, QPen
from iolite.QtGui import QHeaderView, QSplitter, QPlainTextEdit, QDialogButtonBox, QMessageBox, QTableWidget, QTableWidgetItem
from iolite.QtGui import QInputDialog, QFileDialog, QFont, QBrush, QAction, QStyledItemDelegate, QFrame, QFormLayout, QSpinBox, QListWidget
from iolite.QtGui import QItemSelectionModel, QPushButton, QApplication, QMargins, QCompleter
from iolite.QtCore import Signal, QAbstractListModel, QModelIndex, QPoint, QFile, QIODevice, QTimer, QEvent, QSettings, QDir, QRegularExpression
from iolite.Qt import Qt
from iolite.QtUiTools import QUiLoader

from iolite.ui import IolitePlotPyInterface as Plot
from iolite.ui import Iolite3DPlotPyInterface as Plot3d
from iolite.ui import OverlayButton, QCPColorGradient, QCPErrorBars, PythonSyntaxHighlighter, OverlayMessage, QCPRange
from iolite.ui import CommonUIPyInterface as CUI
from iolite.types import Result

import numpy as np
import pandas as pd
import re
import sys
import itertools
import ast

from datetime import datetime
from math import sqrt, log10, ceil, pi as PI
from functools import partial
from types import SimpleNamespace
from enum import Flag, auto

from scipy.interpolate import UnivariateSpline
from scipy.odr import Model, RealData, ODR
from scipy import stats
from scipy.signal import savgol_filter

import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from iolite_helpers import fitLine, formatResult

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Replace this with one of the colorful QCPColorGradients?
import random
colors = {
    i: QColor(random.randrange(50, 200), random.randrange(50, 200), random.randrange(50, 200), 125)
    for i in range(100)
}

'''
Custom Exceptions to help track down issues
'''
class NoRMSelectionsError(RuntimeError):
    pass


class MissingRMGroupError(RuntimeError):
    pass


'''
Block functions and classes
'''

def linear(B, x):
    return B[0]*x + B[1]


def lineartz(B, x):
    return B[0]*x


def makeBeamSeconds():
    method = drs.setting('BeamSecondsMethod')
    channelName = drs.setting('BeamSecondsChannel')
    value = drs.setting('BeamSecondsValue')

    try:
        bs = data.timeSeries('BeamSeconds')
        if bs.property('BeamSecondsMethod') == method and bs.property('BeamSecondsChannel') == channelName and bs.property('BeamSecondsValue') == value:
            return
    except Exception as e:
        print(e)
        pass

    if 'log' in method.lower():
        drs.createBeamSecondsFromLaserLog()
    elif 'gap' in method.lower():
        drs.createBeamSecondsFromSamples()
    elif 'cutoff' in method.lower():
        drs.createBeamSecondsFromCutoff(data.timeSeries(channelName), value)
    elif 'jump' in method.lower():
        drs.createBeamSecondsFromJump(data.timeSeries(channelName), value)
    else:
        print('Tried to make beam seconds with an unknown method')

    data.timeSeries('BeamSeconds').setProperty('BeamSecondsMethod', method)
    data.timeSeries('BeamSeconds').setProperty('BeamSecondsChannel', channelName)
    data.timeSeries('BeamSeconds').setProperty('BeamSecondsValue', value)


class Block(object):

    def __init__(self, selections, label='Unlabeled'):
        self.selections = selections
        self.label = label
        self.fits = {}
        self.lastDFHash = 0
        self.df = None

    def hash(self):
        shash = np.sum([s.midTimestamp for s in self.selections])
        thash = np.sum([c.hash() for c in data.timeSeriesList(data.Intermediate, {'DRSType': 'BaselineSubtracted'}) if 'TotalBeam' not in c.name])
        drshash = drs.setting('NormalizeExternals') + np.sum([ord(a) for a in drs.setting('MasterExternal')]) + np.sum([ord(a) for a in drs.setting('StatName')]) + drs.setting('UseFG')
        return shash+thash+drshash

    def fitHash(self, channel):
        c = data.timeSeries(channel)
        ftz = 0 if not c.property('FitThroughZero') else c.property('FitThroughZero')
        model = np.sum([ord(a) for a in c.property('Model')])
        return self.hash() + ftz + model

    def midTime(self):
        return np.mean([s.midTimestamp if not s.isLinked() else s.linkedMidTimestamp() for s in self.selections])

    def dataFrame(self):
        if self.df is None or self.lastDFHash != self.hash():
            self.updateDataFrame()

        return self.df

    def dataFrameForChannel(self, name):
        df = self.dataFrame()
        cols = [col for col in self.df.columns if col.startswith(name)]
        cols += ['sel_mid_time', 'sel_duration', 'group']
        df = df[cols]
        ext = data.timeSeries(name).property('External standard')
        if type(ext) == str :
            df = df[df['group'].isin(ext.split(','))]
        else:
            raise RuntimeError(f'No externals set for {name}')

        return df

    def updateDataFrame(self):
        self.df = pd.DataFrame()
        ryields, abdf = calculateRelativeYields()

        channelNames = [n for n in data.timeSeriesNames(data.Input) if 'TotalBeam' not in n]
        cpsChannels = [data.timeSeries(f'{c}_CPS') for c in channelNames]

        stat_name = drs.setting('StatName')
        use_fg = drs.setting('UseFG')
        use_isoConcs = drs.setting('UseIsotopicConcentrations')

        #print(f"Using isotopic concentrations? {use_isoConcs}")

        temp_df = data.frame(self.selections, ['UUID', 'group name', 'mid time', 'duration'], cpsChannels, [stat_name, 'int2se'])

        if temp_df is None:
            print(f'### Could not get frame...')
            print(channelNames)
            print(cpsChannels)
            for sel in self.selections:
                print(sel.name)

        temp_df = temp_df.set_index('UUID')
        temp_df = temp_df.rename(columns={
            'group name': 'group',
            'mid time': 'sel_mid_time',
            'duration': 'sel_duration'
        })
        temp_df = temp_df.rename(columns={f'{c}_CPS {stat_name}': f'{c}' for c in channelNames})
        temp_df = temp_df.rename(columns={f'{c}_CPS int2se': f'{c}_Uncert' for c in channelNames})

        groups = temp_df['group'].unique()

        def uncertToUse(rmValue, rmUncert):
            if rmUncert > 0:
                return rmUncert

            if rmValue > 0:
                return rmValue*0.02

            return 1e-6

        for group in groups:
            norm = ryields[group] if drs.setting('NormalizeExternals') else 1
            rmdata = data.referenceMaterialData(group)

            for name, channel in zip(channelNames, cpsChannels):
                try:
                    temp_df.loc[temp_df['group'] == group, name] /= norm
                    temp_df.loc[temp_df['group'] == group, f'{name}_Uncert'] /= norm
                    try:
                        if use_isoConcs:
                            rmd = rmdata[name.replace('_CPS', '')]
                            rmValue = rmd.valueInPPM()
                        else:
                            rmd = rmdata[channel.property('Element')]
                            rmValue = rmd.valueInUnits('fg') if use_fg else rmdata[channel.property('Element')].valueInPPM()

                        rmUncert = rmd.uncertainty()
                    except KeyError as e:
                        # print(f'There was no RM value for {e} for \'{group}\'')
                        continue

                    temp_df.loc[temp_df['group'] == group, f'{name}_RMppm'] = rmValue
                    temp_df.loc[temp_df['group'] == group, f'{name}_RMppm_Uncert'] = uncertToUse(rmValue, rmUncert)

                except Exception as e:
                    print(f'DataFrame Creation Exception: {e}')
                    continue

        if len(temp_df.index) < 1:
            print('DF had zero rows:')
            print(df)

        self.df = temp_df
        self.df = self.df.sort_values(by=['sel_mid_time'])
        self.lastDFHash = self.hash()

    def fit(self, name):
        if data.timeSeries(name).property('External standard') == 'Model':
            # Note: when the property is set with a dict from python
            # it is converted to QVariantMap, therefore the keys become strings
            # so need to use the string version of label to have it match!
            try:
                sens = data.timeSeries(name).property('ModelSensitivities')[str(self.label)]
                self.fits[name] = {
                    'slope': sens,
                    'slope_uncert': sens*0.01,
                    'intercept': 0.,
                    'intercept_uncert': 0.,
                    'hash': self.fitHash(name),
                    'sm_res': 0
                }
            except:
                self.fits[name] = {
                    'slope': np.nan,
                    'slope_uncert': np.nan,
                    'intercept': np.nan,
                    'intercept_uncert': np.nan,
                    'hash': self.fitHash(name),
                    'sm_res': np.nan
                }
        elif name not in self.fits:
            self.fits[name] = self.updateFit(name)

        elif not self.fits[name] is None and self.fits[name]['hash'] != self.fitHash(name):
            self.fits[name] = self.updateFit(name)

        return self.fits[name]

    def updateFit(self, name):
        channel = data.timeSeries(name)
        fitThroughZero = channel.property('FitThroughZero')
        model = channel.property('Model') if channel.property('Model') else 'ODR'
        df = self.dataFrameForChannel(name).copy()

        bad_return = {
            'slope': np.nan,
            'slope_uncert': np.nan,
            'intercept': 0. if fitThroughZero else np.nan,
            'intercept_uncert': 0. if fitThroughZero else np.nan,
            'hash': self.fitHash(name),
            'sm_res': None
        }

        if f'{channel.name}_RMppm' not in df.columns:
            print(f'{channel.name}_RMppm was not in df columns... ')
            # print(df.to_string())
            return bad_return

        if not df[f'{channel.name}_RMppm'].notna().values.any():
            print(f'No data for {channel.name} for chosen RMs in Block {self.label}. No fit.')
            return bad_return

        df = df.dropna()
        func = lineartz if fitThroughZero or len(df['group'].unique()) == 1 else linear
        if len(df['group'].unique()) == 1:
            try:
                slope = df[name].mean()/df[f'{name}_RMppm'].mean()
            except ZeroDivisionError:
                print(f'Replaced slope with 0 for {name} due to a ZeroDivisionError')
                slope = 0.

            try:
                slope_uncert = (df[name].mean()/df[f'{name}_RMppm'].mean())*(df[f'{name}_Uncert'].mean()/df[name].mean())
            except ZeroDivisionError:
                print(f'Replaced slope uncertainty with 0 for {name} due to a ZeroDivisionError')
                slope_uncert = 0.

            intercept = 0.
            intercept_uncert = 0.
            res = None
        else:
            smy = df[f'{name}']
            smx = df[f'{name}_RMppm'] if fitThroughZero else sm.add_constant(df[f'{name}_RMppm'], prepend=False)
            smw = 1/(df[f'{name}_Uncert']/df[name])**2

            if model == 'WLS':
                res = sm.WLS(smy, smx, weights=smw).fit()
            elif model == 'OLS':
                res = sm.OLS(smy, smx).fit()
            elif model == 'RLM':
                res = sm.RLM(smy, smx).fit()
            elif model == 'ODR':
                m = Model(func)
                md = RealData(df[f'{name}_RMppm'], df[f'{name}'], sx=df[f'{name}_RMppm_Uncert'], sy=df[f'{name}_Uncert'])
                b0 = [1000.] if fitThroughZero else [1000., 100.]
                od = ODR(md, m, beta0=b0)
                oo = od.run()
                #oo.pprint()
                res = SimpleNamespace()
                res.params = [oo.beta[0]] if fitThroughZero else [oo.beta[0], oo.beta[1]]
                res.bse = [oo.sd_beta[0]] if fitThroughZero else [oo.sd_beta[0], oo.sd_beta[1]]
            elif model == 'York':
                dfForFit = df
                # If fit is supposed to go through zero, we need to add a data point at 0 for our fitLine routine
                if fitThroughZero:
                    row = {f'{name}_RMppm': 0.0, f'{name}_RMppm_Uncert': 1.0, f'{name}': 0.0, f'{name}_Uncert': 1.0  }
                    dfForFit = df.append(row, ignore_index=True)
                fit = fitLine(dfForFit[f'{name}_RMppm'], dfForFit[f'{name}_RMppm_Uncert'], dfForFit[f'{name}'], dfForFit[f'{name}_Uncert'], np.zeros(len(dfForFit[name])))
                res = SimpleNamespace()
                res.params = [fit['m'], fit['b']]
                res.bse = [fit['sigma_m'], fit['sigma_b']]

            slope = res.params[0]
            slope_uncert = res.bse[0]

            if fitThroughZero:
                intercept = 0.
                intercept_uncert = 0.
            else:
                intercept = res.params[1]
                intercept_uncert = res.bse[1]

        return {
            'slope': slope,
            'slope_uncert': slope_uncert,
            'intercept': 0. if fitThroughZero else intercept,
            'intercept_uncert': 0. if fitThroughZero else intercept_uncert,
            'hash': self.fitHash(name),
            'sm_res': res
        }

    def slope(self, name):
        if not self.fit(name) is None:
            return self.fit(name)['slope']
        else:
            print(f'No fit for {name} in block {self.label}')
            return None

    def slopeUncert(self, name):
        return self.fit(name)['slope_uncert']

    def intercept(self, name):
        return self.fit(name)['intercept']

    def interceptUncert(self, name):
        return self.fit(name)['intercept_uncert']


def calculateRelativeYields():
    groupNames = data.selectionGroupNames(data.ReferenceMaterial)
    masterGroupName = drs.setting('MasterExternal')
    if not masterGroupName:
        masterGroupName = groupNames[0]

    try:
        masterGroup = data.selectionGroup(masterGroupName)
    except Exception as e:
        print(f'Could not find master group {masterGroupName} for relative yield calculation')
        return None, None

    cpsChannelNames = [n for n in data.timeSeriesNames(data.Intermediate, {'DRSType': 'BaselineSubtracted'}) if 'TotalBeam' not in n]
    cpsChannels = [data.timeSeries(c) for c in cpsChannelNames]

    M = np.empty( (len(cpsChannels), len(groupNames)) )
    M.fill(np.nan)

    use_fg = drs.setting('UseFG')
    use_isoConcs = drs.setting('UseIsotopicConcentrations')

    for col, groupName in enumerate(groupNames):
        group = data.selectionGroup(groupName)
        for row, channel in enumerate(cpsChannels):
            channelElement = channel.property('Element')
            try:
                if use_isoConcs:
                    groupRMValue = data.referenceMaterialData(groupName)[channel.name.replace("_CPS", "")].valueInPPM()
                    masterGroupRMValue = data.referenceMaterialData(masterGroupName)[channel.name.replace("_CPS", "")].valueInPPM()

                else:
                    groupRMValue = data.referenceMaterialData(groupName)[channelElement].valueInUnits('fg') if use_fg else data.referenceMaterialData(groupName)[channelElement].valueInPPM()
                    masterGroupRMValue = data.referenceMaterialData(masterGroupName)[channelElement].valueInUnits('fg') if use_fg else data.referenceMaterialData(masterGroupName)[channelElement].valueInPPM()

                groupResult = data.groupResult(group, channel).value()
                masterGroupResult = data.groupResult(masterGroup, channel).value()
                M[row][col] = (groupResult/groupRMValue) / (masterGroupResult/masterGroupRMValue)
            except Exception as e:
                pass
                #print(f'Problem: {group.name} {channelElement} {groupResult} {groupRMValue} {masterGroupResult} {masterGroupRMValue}')

    # Todo: investigate whether there are any trends with the relative yield and mass or Tc?
    df = pd.DataFrame(M, columns=groupNames)
    #df['Tc'] = [data.elements[c.property('Element')]['Tcond_Lodders'] for c in cpsChannels]
    df['Element'] = [c.property('Element') for c in cpsChannels]
    ablationFactors = dict(zip(groupNames, np.nanmedian(M, axis=0)))
    return ablationFactors, df

def assignExternalAffinities(sels=None):
    from sklearn.neighbors import NearestCentroid
    from sklearn.preprocessing import StandardScaler

    #externalsInUse = list(set(list(itertools.chain(*[c.property('External standard').split(',') for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name and 'External standard' in c.properties().keys()]))))
    externalsInUse = drs.externalsInUse()

    if sels is None:
        sels = list(itertools.chain(*[sg.selections() for sg in data.selectionGroupList(data.ReferenceMaterial | data.Sample)]))

    if len(externalsInUse) == 0:
        return externalsInUse
    if len(externalsInUse) == 1:
        [s.setProperty('External affinity', externalsInUse[0]) for s in sels]
        return externalsInUse

    isElements = list(set([s.property('Internal element') for s in sels]))
    affinityElements = list(set([s.property('Affinity elements') for s in sels]))

    #print(f'assignExternalAffinities with {isElements} {affinityElements}')

    if len(isElements) == 0 or len(affinityElements) == 0:
        print('Could not update external affinities for selections...')
        return externalsInUse

    combinations = list(itertools.product(isElements, affinityElements))
    #print(f'Combinations are {combinations}')

    def sumForSelection(sel, channels):
        n = len(data.timeSeries(channels[0]).dataForSelection(sel))
        s = np.zeros(n)
        for c in channels:
            s += data.timeSeries(f'{c}_CPS').dataForSelection(sel)

        return s

    aff = {}
    for comb in combinations:
        #print(comb)
        ise = comb[0]
        afe = comb[1]
        if not ise or not afe or afe in externalsInUse:
            print(f"Continuing due to unset IntStd or Affinity not being specified")
            continue

        X = np.empty( (0, len(afe.split(','))) )
        y = np.array([])
        for gi, gn in enumerate(externalsInUse):
            g = data.selectionGroup(gn)
            for sel in g.selections():
                norm = sumForSelection(sel, ise.split(','))
                sd = np.column_stack( (data.timeSeries(f'{c}_CPS').dataForSelection(sel)/norm for c in afe.split(',')) )
                X = np.vstack( (X, sd) )
                gy = np.array([gi]*len(norm))
                y = np.concatenate( (y, gy) )

        scaler = StandardScaler().fit(X, y)
        X = scaler.transform(X)

        nc = NearestCentroid()
        nc.fit(X,y)
        aff[comb] = {'classifier': nc, 'scaler': scaler}

    for sel in sels:
        ise = sel.property('Internal element')
        afe = sel.property('Affinity elements')
        if not ise or not afe:
            continue

        if afe in externalsInUse:
            # If a selection has a rm for affinity elements rather than a list of elements
            # set it to have that external affinity without using the classifier.
            sel.setProperty('External affinity', afe)
        else:
            channels = afe.split(',')
            norm = sumForSelection(sel, ise.split(','))
            seld = np.array([np.median(data.timeSeries(f'{c}_CPS').dataForSelection(sel)/norm) for c in channels]).reshape(1, -1)
            seld = np.nan_to_num(aff[(ise, afe)]['scaler'].transform(seld))
            i = int(aff[(ise,afe)]['classifier'].predict(seld)[0])
            sel.setProperty('External affinity', externalsInUse[i])

    return externalsInUse


def findBlocks(method=None):
    '''
    Have 4 modes:
        1. Assigned - Only use selections that have been assigned explicitly (fastest)
        2. Simple - Use the fast method looking at selection time jumps to find new blocks (faster)
        3. Clustering - Use clustering with a specific number of blocks (fast)
        4. Auto Clustering - Use clustering that searches for the *best* number of blocks (slow)

    We also want to be able to use one of the more automatic methods with some (not all) selections being specified.
    '''
    externalsInUse = set(list(itertools.chain(*[c.property('External standard').split(',') for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name and type(c.property('External standard')) == str])))
    try:
        externalsInUse.remove('Model')
    except:
        pass

    # the user may have selected an external that doesn't exist. If so, it will raise a RuntimeError.
    # That exception should be caught in calling functions
    try:
        groups = [data.selectionGroup(ext) for ext in externalsInUse if ext]
    except RuntimeError:
        raise MissingRMGroupError

    selections = list(itertools.chain(*[sg.selections() for sg in groups]))

    if len(selections) == 0:
        print('No selections to find blocks for!')
        raise MissingRMGroupError

    # Create a list of component selections
    component_sels = list(itertools.chain(*[s.linkedSelections() for s in selections if s.hasLinks()]))

    # Now kick them out
    selections = list(filter(lambda s: s.property('UUID') not in component_sels, selections))

    def sel_sorter(sel):
        if not sel.isLinked():
            return sel.midTimestamp
        else:
            return sel.linkedMidTimestamp()

    selections.sort(key=sel_sorter)
    selMidTimes = [s.midTimestamp if not s.isLinked() else s.linkedMidTimestamp() for s in selections]

    def assignedBlock(s):
        if s.property('Block') is not None:
            return s.property('Block')

        return -1

    assignedBlocks = np.array([assignedBlock(s) for s in selections])
    allAssigned = np.all(assignedBlocks>=0)

    diffs = np.column_stack( (range(len(selMidTimes)), np.insert(np.diff(selMidTimes), 0, 0)))
    cutoff = np.mean(diffs[:, 1])

    if not method:
        method = 'Simple'

    if method == 'Simple' and not allAssigned:
        current_label = 1
        labels = []
        for i, v in enumerate(diffs[:, 1]):
            if v > cutoff:
                current_label += 1
            labels.append(current_label)
    elif method == 'Auto Clustering' and not allAssigned:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        scores = []
        a = np.array(selMidTimes).reshape(-1,1)
        for nc in range(2, len(selMidTimes)):
            km = KMeans(n_clusters=nc).fit(a)
            ss = silhouette_score(a, km.labels_)
            scores.append(ss)

        nc = np.argmax(scores)+2 # 1 (bc above starts at nc=2) + 1 (because starting at 0)
        print(f'findBlocks decided to use {nc} clusters')
        km = KMeans(n_clusters=int(nc)).fit(a)
        labels = km.labels_ + 1
    elif method == 'Clustering' and not allAssigned:
        from sklearn.cluster import KMeans
        nc = drs.setting('NClusters')
        if not nc:
            nc = 5
        print(f'findBlocks decided to use {nc} clusters')
        a = np.array(selMidTimes).reshape(-1,1)
        km = KMeans(n_clusters=int(nc)).fit(a)
        labels = km.labels_ + 1
    else:
        labels = assignedBlocks

    block_selections = {}

    #print(f'Used method = {method}, got labels = {labels}')
    specified = [si for si, s in enumerate(selections) if s.property('Block') is not None]
    for k in np.unique(labels):
        matches = np.where(np.array(labels).astype(int) == k)[0]
        ind = [i for i in matches if i not in specified]
        block_selections[k] = list(np.array(selections)[ind])

    blocks = []

    for k in block_selections:
        specified = [s for si, s in enumerate(selections) if s.property('Block') == k]
        if len(block_selections[k] + specified) == 0:
            continue
        blocks.append(Block(block_selections[k] + specified, k))

    blocks.sort(key=lambda b: b.midTime())
    # Relabel according to time order:
    for i, block in enumerate(blocks):
        block.label = i+1

    return blocks


def fitSurface(blocks, channelName):

    slopes = np.array([block.slope(channelName) for block in blocks])
    slopes_unc = np.array([block.slopeUncert(channelName) for block in blocks])
    intercepts = np.array([block.intercept(channelName) for block in blocks])
    intercepts_unc = np.array([block.interceptUncert(channelName) for block in blocks])
    times = np.array([block.midTime() for block in blocks])

    # Drop nan values... this allows having some blocks without a particular element
    nan_locs = np.union1d(np.argwhere(np.isnan(slopes)), np.argwhere(np.isnan(intercepts)))
    slopes = np.delete(slopes, nan_locs)
    slopes_unc = np.delete(slopes_unc, nan_locs)
    intercepts = np.delete(intercepts, nan_locs)
    intercepts_unc = np.delete(intercepts_unc, nan_locs)
    times = np.delete(times, nan_locs)

    cpsChannel = data.timeSeries(f'{channelName}_CPS')
    splineType = drs.setting('SplineType')

    oneBlockTypes = ['MeanMean', 'MeanMedian']
    lowBlockCountTypes = ['MeanMean', 'MeanMedian', 'LinearFit', 'WeightedLinearFit', 'StepLinear','StepForward', 'StepBackward', 'StepAverage', 'Nearest']

    if len(slopes) < 2 and not splineType in oneBlockTypes:
        #IoLog.information("Changing spline type to MeanMean due to fewer than 2 blocks found")
        splineType = 'MeanMean'
    elif len(slopes) < 5 and not splineType in lowBlockCountTypes:
        #IoLog.information("Changing spline type to StepLinear due to fewer than 5 blocks found")
        splineType = 'StepLinear'

    if np.all(np.isnan(slopes)):
        return None, None

    slopes_unc[slopes_unc == 0] = np.nanmean(slopes)*0.05
    intercepts_unc[intercepts_unc == 0] = 1e-6

    slope_spl = data.spline(times, slopes, slopes_unc, splineType, cpsChannel.time())
    intercept_spl = data.spline(times, intercepts, intercepts_unc, splineType, cpsChannel.time())

    data.createTimeSeries(f'{channelName}_slope', data.Intermediate, cpsChannel.time(), slope_spl, {'DRS': '3D Trace Elements'})
    data.createTimeSeries(f'{channelName}_intercept', data.Intermediate, cpsChannel.time(), intercept_spl, {'DRS': '3D Trace Elements'})

    def surface(t, c):
        if len(t) != len(cpsChannel.time()):
            i = np.searchsorted(cpsChannel.time(), t)
            m = slope_spl[i]
            b = intercept_spl[i]
            return m*c + b
        else:
            return slope_spl*c + intercept_spl


    def surfaceInv(t, I):
        if len(t) != len(cpsChannel.time()):
            i = np.searchsorted(cpsChannel.time(), t)
            m = slope_spl[i]
            b = intercept_spl[i]
            # I = m*c + b, so c = (I - b)/m
            return (I - b)/m
        else:
            r = (I - intercept_spl)/slope_spl
            r[np.isinf(r)] = np.nan
            return r

    return surface, surfaceInv


class Calibration(object):

    normal = 0
    inverse = 1

    def __init__(self):
        self.surfaces = {}
        self.blocks = []
        self.frac = {}

    def block(self, bn):
        if bn >= len(self.blocks):
            raise RuntimeError(f'Block number {bn} out of range {len(self.blocks)}')

        return self.blocks[bn]

    def updateBlocks(self):
        self.blocks = findBlocks(drs.setting('BlockFindingMethod'))

    def surface(self, name, update=False, inv=False):
        if name not in self.surfaces or update:
            self.updateSurface(name)

        s = self.surfaces[name][self.inverse] if inv else self.surfaces[name][self.normal]
        return s

    def updateSurface(self, name):
        self.surfaces[name] = fitSurface(self.blocks, name)

    def semiquant(self, name):
        cps = data.timeSeries(f'{name}_CPS')
        return self.surface(name, inv=True)(cps.time(), cps.data())

    def clearFractionationCache(self):
        self.frac = {}

    def clearSurfaceCache(self):
        self.surfaces = {}

    def fractionation(self, name, update=False):
        if name not in self.frac or update:
            print(f"Updating fractionation for {name}")
            self.updateFractionation(name)

        return self.frac[name]

    def updateFractionation(self, name):
        channel = data.timeSeries(name)

        try:
            externals = [es for es in channel.property('External standard').split(',') if es]
        except:
            self.frac[name] = pd.DataFrame()
            return

        if not externals:
            self.frac[name] = pd.DataFrame()
            return

        try:
            groups = [data.selectionGroup(ext) for ext in externals]
        except RuntimeError:
            print(f'Could not find external group for fractionation calculation for {name}')
            self.frac[name] = pd.DataFrame()
            return

        cpsChannel = data.timeSeries(f'{name}_CPS')

        try:
            makeBeamSeconds()
            bs = data.timeSeries('BeamSeconds')
        except Exception as e:
            print(e)
            print('There was a problem making beam seconds and therefore fractionation cannot be determined.')
            return

        allSels = [[s for s in sg.selections()] for sg in data.selectionGroupList(data.ReferenceMaterial | data.Sample)]
        allSels = list(itertools.chain.from_iterable(allSels))
        allIS = [sel.property('Internal element') for sel in allSels]
        isElementsList = [ie for ie in set(allIS) if ie != 'None' and ie != '' and ie]
        isElementsList.sort()

        use_fg = drs.setting('UseFG')

        def sumsForSel(sel, isElements):
            if not isElements:
                raise RuntimeError('No internals set')

            selInd = cpsChannel.selectionIndices(sel)
            selPPMSum = np.zeros(len(bs.timeForSelection(sel)))
            rmPPMSum = 0
            for el in [el for el in isElements.split(',') if el]:
                selPPMSum += self.semiquant(el)[selInd]
                rmPPMSum += data.referenceMaterialData(sel.group().name)[data.timeSeries(f'{el}_CPS').property('Element')].valueInUnits('fg') if use_fg else data.referenceMaterialData(sel.group().name)[data.timeSeries(f'{el}_CPS').property('Element')].valueInPPM()
            return selPPMSum, rmPPMSum

        fdf = pd.DataFrame()

        for sg in groups: # Loop through each external for this channel
            for sel in sg.selections(): # Loop through each selection of each group
                for isElements in isElementsList: # Collect ratio data for each of the IS combinations in use
                    try:
                        selPPMSum, rmPPMSum = sumsForSel(sel, isElements)
                        thisPPM = data.referenceMaterialData(sg.name)[cpsChannel.property('Element')].valueInUnits('fg') if use_fg else data.referenceMaterialData(sg.name)[cpsChannel.property('Element')].valueInPPM()

                        t = bs.dataForSelection(sel)
                        selInd = cpsChannel.selectionIndices(sel)
                        sq = self.semiquant(name)[selInd]
                        r = (sq/selPPMSum)*(rmPPMSum/thisPPM)

                        if isElements == name:
                            r = np.ones(len(r))

                        df = pd.DataFrame({'t': t, 'r': r, 'IS': isElements, 'group': sg.name}, index=pd.Series([pd.Timedelta(milliseconds=int(tt*1000)) for tt in t]))
                        fdf = fdf.append(df) if len(fdf) > 0 else df
                    except Exception as e:
                        continue

        self.frac[name] = fdf

    def fitFractionation(self, name, isElements=None, td=None, k=None, group=None):
        #print("Fitting fractionation....")
        ft = data.timeSeries(name).property('FractionationFitType')
        fc = data.timeSeries(name).property('FractionationCorrection')

        if not k:
            k = 1 if not ft or ft == 'Linear' else 3

        if not isElements:
            isdf = self.fractionation(name)
        else:
            fdf = self.fractionation(name)
            isdf = fdf[fdf['IS'] == isElements]

        if group is not None:
            isdf = isdf[isdf['group'] == group]

        if not td:
            # Aim for 30 points
            td = (isdf.index.max()-isdf.index.min())/30
        elif isinstance(td, str):
            td = pd.Timedelta(td)

        if not td or len(isdf) == 0:
            print(f'Could not fit fractionation for {name} {isElements} {td} {k} {group}')
            return None, None, None, None

        rsd = isdf.resample(td).sem()['r']

        isdf = isdf.resample(td).median()
        t = isdf['t']
        r = isdf['r']

        rsd[rsd==0] = 0.02*np.nanmean(r)
        rsd[rsd!=rsd] = 0.02*np.nanmean(r)
        rsd[rsd<0.001*np.nanmean(r)] = 0.02*np.nanmean(r)
        rsd[rsd>1] = 0.02*np.nanmean(r)

        def ones(x):
            try:
                return np.ones(len(x))
            except:
                return np.ones(1)

        if fc and name != isElements:
            sx = np.linspace(0, t.max(), 100)
            if k == 3:
                spline = UnivariateSpline(t[1:-1], r[1:-1], w=1/rsd[1:-1], k=k, s=len(t)*2)
            else:
                spline = UnivariateSpline(t[1:-1], r[1:-1], w=1/rsd[1:-1], k=k, s=1e9)
        elif fc and name == isElements:
            spline = ones
        else:
            spline = None

        return t, r, rsd, spline

    def uncertainty(self, name):
        # There is a (small) chance that there may be zero counts for our RM
        # for a particular channel. If we only have one RM, this will result in
        # a slope of 0, and we'll get  a divide by zero exception below....
        # Just going to replace any 0 slopes with very small numbers below...
        slopes = [block.slope(name) for block in self.blocks]
        for i, s in enumerate(slopes):
            if s == 0:
                slopes[i] = 0.00000001
                print(f'Replaced slope for Block {i} for {name}')

        uncerts = [block.slopeUncert(name) for block in self.blocks]
        return np.mean([uncert / slope for uncert, slope in zip(uncerts, slopes)])
        # return np.mean([block.slopeUncert(name)/block.slope(name) for block in self.blocks])

#    def materialFractionation(self, selection, masterExt):
#        '''
#        Want to calculate:
#            (master_IS_SQ/selection_channel_SQ) * (selection_IS_ppm / master_IS_ppm)
#        '''

#        try:
#            bs = data.timeSeries('BeamSeconds')
#        except:
#            drs.createBeamSecondsFromLaserLog()
#            bs = data.timeSeries('BeamSeconds')

#        isChannels = selection.property('Internal element').split(',')
#        isValue = selection.property('Internal value') # TODO: get units and convert this to ppm if it isn't
#        t = data.timeSeries('TotalBeam').time()

#        mdf = pd.DataFrame()
#        masterPPM = np.sum([data.referenceMaterialData(masterExt)[data.timeSeries(channelName).property('Element')].valueInPPM() for channelName in isChannels])

#        sq = {}
#        for channelName in isChannels:
#            sq[channelName] = self.semiquant(channelName)

#        for channelName in isChannels:
#            dht, dhd = data.compileDownholeFromArray(data.selectionGroup(masterExt), sq[channelName])
#            df = pd.DataFrame({'t': dht, f'Master_{channelName}':dhd}, index=pd.Series([pd.Timedelta(milliseconds=int(tt*1000)) for tt in dht]))
#            mdf = mdf.append(df) if len(mdf) > 0 else df

#        selectionSQ = np.zeros(len(t))
#        ind = data.timeSeries('TotalBeam').selectionIndices(selection)
#        selt = data.timeSeries('BeamSeconds').data()[ind]
#        for channelName in isChannels:
#            df = pd.DataFrame({'t': selt, f'Selection_{channelName}': sq[channelName][ind]}, index=pd.Series([pd.Timedelta(milliseconds=int(tt*1000)) for tt in selt]))
#            mdf = mdf.append(df)


#        td = (mdf.index.max()-mdf.index.min())/30
#        sdf = mdf.resample(td).median()
#        sdf = sdf[1:-1]
#        sdf = sdf.dropna()

#        masterSQ = sdf[ [f'Master_{channelName}' for channelName in isChannels]].sum(axis=1)
#        selectionSQ = sdf[ [f'Selection_{channelName}' for channelName in isChannels]].sum(axis=1)
#        t = sdf['t']
#        #fd = (masterSQ/selectionSQ) * (isValue/masterPPM)
#        fd = (selectionSQ/masterSQ) * (masterPPM/isValue)

#        fit = np.polyfit(t, fd, 1)
#        return (t, fd), partial(linear, (fit[0], fit[1]))

#        #return (masterSQ/selectionSQ) * (isValue/masterPPM)


def runDRS():
    drs.message("Starting 3D Trace Elements DRS...")
    drs.progress(0)
    drs.setProperty('isRunning', True)
    commonProps = {'DRS': drs.name()}

    # Get settings
    settings = drs.settings()

    print(settings)

    # If running as a processing template, need to copy over defaults and
    # other custom externals/internals to the actual channels available
    if drs.isPTAction():
        print('Applying PT options...')
        if 'IndexChannel' not in settings or settings['IndexChannel'] == '':
            settings['IndexChannel'] = data.timeSeriesNames(data.Input)[0]

        pt_exts = settings['PTExternal']
        for ext in pt_exts:
            channels = [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]
            if ext['name'] != 'Default':
                channels = [c for c in channels if len(re.findall(ext['name'], c.name)) > 0]

            for channel in channels:
                channel.setProperty('External standard', ext['standards'])
                channel.setProperty('FitThroughZero', ext['zero'])
                channel.setProperty('Model', ext['model'])
                if ext['frac'] == 'None':
                    channel.setProperty('FractionationCorrection', False)
                else:
                    channel.setProperty('FractionationCorrection', True)
                    channel.setProperty('FractionationFitType', ext['frac'])

        pt_is = None
        if 'PTInternal' in settings.keys():
            pt_is = settings['PTInternal']
        requiredProperties = ['Internal element', 'Internal value', 'Internal units']

        if pt_is is None:
            pt_is = []
        for intstd in pt_is:
            sels = list(itertools.chain(*[sg.selections() for sg in data.selectionGroupList(data.ReferenceMaterial | data.Sample)]))
            if intstd['group'] != 'Default':
                sels = [s for s in sels if len(re.findall(intstd['group'], s.group().name)) > 0]

            if intstd['selection'] != 'Default':
                sels = [s for s in sels if len(re.findall(intstd['selection'], s.name)) > 0]

            for sel in sels:
                if 'PreserveISProperties' in settings and settings['PreserveISProperties'] and set(requiredProperties) <= set(sel.properties().keys()):
                    continue
                sel.setProperty('Internal element', intstd['elements'])
                sel.setProperty('Internal value', intstd['value'])
                sel.setProperty('Internal units', intstd['units'])
                sel.setProperty('External affinity', intstd['affinity'])


    drs.clearSelectionProperties(QRegularExpression("Sensitivity.+"))
    drs.clearSelectionProperties(QRegularExpression("External Sensitivity.+"))

    indexChannel = data.timeSeries(settings["IndexChannel"])
    drs.setIndexChannel(indexChannel)

    # Setup index time
    drs.message("Setting up index time...")
    drs.progress(5)
    drs.setIndexChannel(indexChannel)

    # Check if we need to do baseline subtractions:
    bl_required = not np.all(np.array([bool(ch.property('BackgroundSubtracted')) for ch in data.timeSeriesList(data.Input) if 'TotalBeam' not in ch.name]))

    if bl_required:
        try:
            blGrp = data.selectionGroupList(data.Baseline)[0]
        except:
            IoLog.error("There are no baseline groups. 3D Trace Elements cannot proceed...")
            drs.message("Error. See Messages")
            drs.progress(100)
            drs.finished()
            return

        if len(blGrp.selections()) < 1:
            IoLog.error("No baseline selections. Please select some baselines. 3D Trace Elements cannot proceed...")
            drs.message("Error. See Messages")
            drs.progress(100)
            drs.finished()
            return

    # Setup the mask
    maskOption = settings["Mask"]

    if maskOption:
        drs.message("Making mask...")
        drs.progress(10)
        maskMethod = settings['MaskMethod']
        trim = settings["MaskTrim"]

        if 'Laser' in maskMethod:
            mask = drs.createMaskFromLaserLog(trim)
        else:
            maskChannel = data.timeSeries(settings["MaskChannel"])
            cutoff = settings["MaskCutoff"]
            mask = drs.createMaskFromCutoff(maskChannel, cutoff, trim)
        data.createTimeSeries('mask', data.Intermediate, indexChannel.time(), mask)
    else:
        mask = np.ones_like(indexChannel.data())
        data.createTimeSeries('mask', data.Intermediate, indexChannel.time(), mask)


#---Morgan edit 042925---


    isotope_masses = {
    # Ac
        "Ac227": 227.0277,

    # Ag
        "Ag107": 106.9051,
        "Ag109": 108.9048,

    # Ar
        "Ar36": 35.9675,
        "Ar38": 37.9627,
        "Ar40": 39.9624,

    # As
        "As75": 74.9216,

    # At
        "At209": 208.9876,

    # Au
        "Au197": 196.9666,

    # Ba
        "Ba130": 129.9063,
        "Ba132": 131.9051,
        "Ba134": 133.9045,
        "Ba135": 134.9057,
        "Ba136": 135.9046,
        "Ba137": 136.9058,
        "Ba138": 137.9052,

    # Bi
        "Bi209": 208.9804,

    # Br
        "Br79": 78.9183,
        "Br81": 80.9163,

    # Ca
        "Ca40": 39.9626,
        "Ca42": 41.9586,
        "Ca43": 42.9588,
        "Ca44": 43.9555,
        "Ca46": 45.9537,
        "Ca48": 47.9525,

    # Cd
        "Cd106": 105.9065,
        "Cd108": 107.9042,
        "Cd110": 109.903,
        "Cd111": 110.9042,
        "Cd112": 111.9028,
        "Cd113": 112.9044,
        "Cd114": 113.9034,
        "Cd116": 115.9048,

    # Ce
        "Ce136": 135.9071,
        "Ce138": 137.9059,
        "Ce140": 139.9054,
        "Ce142": 141.9092,

    # Cl
        "Cl35": 34.9689,
        "Cl37": 36.9659,

    # Co
        "Co59": 58.9332,

    # Cr
        "Cr50": 49.946,
        "Cr52": 51.9405,
        "Cr53": 52.9406,
        "Cr54": 53.9389,

    # Cs
        "Cs133": 132.9054,

    # Cu
        "Cu63": 62.9296,
        "Cu65": 64.9278,

    # Dy
        "Dy156": 155.9243,
        "Dy158": 157.9244,
        "Dy160": 159.9252,
        "Dy161": 160.9269,
        "Dy162": 161.9268,
        "Dy163": 162.9287,
        "Dy164": 163.9292,

    # Er
        "Er162": 161.9288,
        "Er164": 163.9292,
        "Er166": 165.9303,
        "Er167": 166.932,
        "Er168": 167.9324,
        "Er170": 169.9355,

    # Eu
        "Eu151": 150.9199,
        "Eu153": 152.9212,

    # Fe
        "Fe54": 53.9396,
        "Fe56": 55.9349,
        "Fe57": 56.9354,
        "Fe58": 57.9333,

    # Fr
        "Fr223": 223.0197,

    # Ga
        "Ga69": 68.9256,
        "Ga71": 70.9247,

    # Gd
        "Gd152": 151.9198,
        "Gd154": 153.9209,
        "Gd155": 154.9226,
        "Gd156": 155.9221,
        "Gd157": 156.9239,
        "Gd158": 157.9241,
        "Gd160": 159.9271,

    # Ge
        "Ge70": 69.9242,
        "Ge72": 71.9221,
        "Ge73": 72.9235,
        "Ge74": 73.9212,
        "Ge76": 75.9214,

    # Hf
        "Hf174": 173.94,
        "Hf176": 175.9414,
        "Hf177": 176.9432,
        "Hf178": 177.9437,
        "Hf179": 178.9458,
        "Hf180": 179.9465,

    # Hg
        "Hg196": 195.9658,
        "Hg198": 197.9667,
        "Hg199": 198.9683,
        "Hg200": 199.9683,
        "Hg201": 200.9703,
        "Hg202": 201.9706,
        "Hg204": 203.9735,

    # Ho
        "Ho165": 164.9303,

    # I
        "I127": 126.9045,

    # In
        "In113": 112.9041,
        "In115": 114.9039,

    # Ir
        "Ir191": 190.9606,
        "Ir193": 192.9629,

    # K
        "K39": 38.9637,
        "K40": 39.9639,
        "K41": 40.9618,

    # Kr
        "Kr78": 77.9204,
        "Kr80": 79.9164,
        "Kr82": 81.9135,
        "Kr83": 82.9141,
        "Kr84": 83.9115,
        "Kr86": 85.9106,

    # La
        "La138": 137.9071,
        "La139": 138.9063,

    # Lu
        "Lu175": 174.9408,
        "Lu176": 175.9427,

    # Mg
        "Mg24": 23.985,
        "Mg25": 24.9858,
        "Mg26": 25.9826,

    # Mo
        "Mo92": 91.9068,
        "Mo94": 93.9051,
        "Mo95": 94.9058,
        "Mo96": 95.9047,
        "Mo97": 96.906,
        "Mo98": 97.9054,
        "Mo100": 99.9075,

    # Mn
        "Mn55": 54.938,

    # Nd
        "Nd142": 141.9077,
        "Nd143": 142.9098,
        "Nd144": 143.9101,
        "Nd145": 144.9126,
        "Nd146": 145.9131,
        "Nd148": 147.9169,
        "Nd150": 149.9209,

    # Nb
        "Nb93": 92.9064,

    # Ni
        "Ni58": 57.9353,
        "Ni60": 59.9308,
        "Ni61": 60.9311,
        "Ni62": 61.9283,
        "Ni64": 63.9279,

    # Os
        "Os184": 183.9525,
        "Os186": 185.9538,
        "Os187": 186.9557,
        "Os188": 187.9558,
        "Os189": 188.9581,
        "Os190": 189.9584,
        "Os192": 191.9615,

    # P
        "P31": 30.9738,

    # Pa
        "Pa231": 231.0359,

    # Pb
        "Pb204": 203.973,
        "Pb206": 205.9745,
        "Pb207": 206.9759,
        "Pb208": 207.9766,

    # Pd
        "Pd102": 101.9056,
        "Pd104": 103.904,
        "Pd105": 104.9051,
        "Pd106": 105.9035,
        "Pd108": 107.9039,
        "Pd110": 109.9052,

    # Po
        "Po208": 207.982,
        "Po209": 208.9824,

    # Pr
        "Pr141": 140.9077,

    # Pt
        "Pt190": 189.9599,
        "Pt192": 191.961,
        "Pt194": 193.9627,
        "Pt195": 194.9648,
        "Pt196": 195.9649,
        "Pt198": 197.9679,

    # Ra
        "Ra226": 226.0254,

    # Rb
        "Rb85": 84.9118,
        "Rb87": 86.9092,

    # Re
        "Re185": 184.9529,
        "Re187": 186.9557,

    # Rh
        "Rh103": 102.9055,

    # Ru
        "Ru96": 95.9076,
        "Ru98": 97.9053,
        "Ru99": 98.9059,
        "Ru100": 99.9042,
        "Ru101": 100.9056,
        "Ru102": 101.9043,
        "Ru104": 103.9054,

    # S
        "S32": 31.9721,
        "S33": 32.9715,
        "S34": 33.9679,
        "S36": 35.9671,

    # Sb
        "Sb121": 120.9038,
        "Sb123": 122.9042,

    # Sc
        "Sc45": 44.9559,

    # Se
        "Se74": 73.9225,
        "Se76": 75.9192,
        "Se77": 76.9199,
        "Se78": 77.9173,
        "Se80": 79.9165,
        "Se82": 81.9167,

    # Si
        "Si28": 27.9769,
        "Si29": 28.9765,
        "Si30": 29.9738,

    # Sm
        "Sm144": 143.912,
        "Sm147": 146.9149,
        "Sm148": 147.9148,
        "Sm149": 148.9172,
        "Sm150": 149.9173,
        "Sm152": 151.9197,
        "Sm154": 153.9222,

    # Sn
        "Sn112": 111.9048,
        "Sn114": 113.9028,
        "Sn115": 114.9033,
        "Sn116": 115.9017,
        "Sn117": 116.9029,
        "Sn118": 117.9016,
        "Sn119": 118.9033,
        "Sn120": 119.9022,
        "Sn122": 121.9034,
        "Sn124": 123.9053,

    # Sr
        "Sr84": 83.9134,
        "Sr86": 85.9093,
        "Sr87": 86.9089,
        "Sr88": 87.9056,

    # Ta
        "Ta180": 179.9475,
        "Ta181": 180.9479,

    # Tb
        "Tb159": 158.9254,

    # Tc
        "Tc97": 96.9064,
        "Tc98": 97.9072,
        "Tc99": 98.9063,

    # Te
        "Te120": 119.904,
        "Te122": 121.903,
        "Te123": 122.9043,
        "Te124": 123.9028,
        "Te125": 124.9044,
        "Te126": 125.9033,
        "Te128": 127.9045,
        "Te130": 129.9062,

    # Th
        "Th232": 232.0381,

    # Ti
        "Ti46": 45.9526,
        "Ti47": 46.9518,
        "Ti48": 47.9479,
        "Ti49": 48.9479,
        "Ti50": 49.9448,

    # Tl
        "Tl203": 202.9723,
        "Tl205": 204.9744,

    # Tm
        "Tm169": 168.9342,

    # U
        "U234": 234.0409,
        "U235": 235.0439,
        "U238": 238.0508,

    # V
        "V50": 49.9472,
        "V51": 50.9439,

    # W
        "W180": 179.9467,
        "W182": 181.9482,
        "W183": 182.9502,
        "W184": 183.9509,
        "W186": 185.9544,

    # Xe
        "Xe124": 123.9059,
        "Xe126": 125.9043,
        "Xe128": 127.9035,
        "Xe129": 128.9048,
        "Xe130": 129.9035,
        "Xe131": 130.9051,
        "Xe132": 131.9042,
        "Xe134": 133.9054,
        "Xe136": 135.9072,

    # Y
        "Y89": 88.9058,

    # Yb
        "Yb168": 167.9339,
        "Yb170": 169.9348,
        "Yb171": 170.9363,
        "Yb172": 171.9364,
        "Yb173": 172.9382,
        "Yb174": 173.9389,
        "Yb176": 175.9426,

    # Zn
        "Zn64": 63.9291,
        "Zn66": 65.926,
        "Zn67": 66.9271,
        "Zn68": 67.9248,
        "Zn70": 69.9253,

    # Zr
        "Zr90": 89.9047,
        "Zr91": 90.9056,
        "Zr92": 91.905,
        "Zr94": 93.9063,
        "Zr96": 95.9083,

    }


    natural_abundances = {
    # Ag
        "Ag107": 0.51839,
        "Ag109": 0.48161,

    # As
        "As75": 1,

    # Au
        "Au197": 1,

    # Ba
        "Ba130": 0.00106,
        "Ba132": 0.00101,
        "Ba134": 0.02417,
        "Ba135": 0.06592,
        "Ba136": 0.07854,
        "Ba137": 0.11232,
        "Ba138": 0.71698,

    # Bi
        "Bi209": 1,

    # Br
        "Br79": 0.5069,
        "Br81": 0.4931,

    # Ca
        "Ca40": 0.96941,
        "Ca42": 0.00647,
        "Ca43": 0.00135,
        "Ca44": 0.02086,
        "Ca46": 4e-05,
        "Ca48": 0.00187,

    # Cd
        "Cd106": 0.0125,
        "Cd108": 0.0089,
        "Cd110": 0.1249,
        "Cd111": 0.128,
        "Cd112": 0.2413,
        "Cd113": 0.1222,
        "Cd114": 0.2873,
        "Cd116": 0.0749,

    # Ce
        "Ce136": 0.00185,
        "Ce138": 0.00251,
        "Ce140": 0.8845,
        "Ce142": 0.11114,

    # Co
        "Co59": 1,

    # Cr
        "Cr50": 0.04345,
        "Cr52": 0.83789,
        "Cr53": 0.09501,
        "Cr54": 0.02365,

    # Cu
        "Cu63": 0.6915,
        "Cu65": 0.3085,

    # Dy
        "Dy156": 0.0006,
        "Dy158": 0.0001,
        "Dy160": 0.0233,
        "Dy161": 0.1889,
        "Dy162": 0.2548,
        "Dy163": 0.2497,
        "Dy164": 0.2818,

    # Er
        "Er162": 0.0014,
        "Er164": 0.0161,
        "Er166": 0.3361,
        "Er167": 0.2293,
        "Er168": 0.2678,
        "Er170": 0.1493,

    # Eu
        "Eu151": 0.4781,
        "Eu153": 0.5219,

    # Fe
        "Fe54": 0.05845,
        "Fe56": 0.91754,
        "Fe57": 0.02119,
        "Fe58": 0.00282,

    # Ga
        "Ga69": 0.60108,
        "Ga71": 0.39892,

    # Gd
        "Gd152": 0.002,
        "Gd154": 0.0218,
        "Gd155": 0.148,
        "Gd156": 0.2047,
        "Gd157": 0.1565,
        "Gd158": 0.2484,
        "Gd160": 0.2186,

    # Ge
        "Ge70": 0.2057,
        "Ge72": 0.2745,
        "Ge73": 0.0775,
        "Ge74": 0.365,
        "Ge76": 0.0773,

    # Hf
        "Hf174": 0.0016,
        "Hf176": 0.0526,
        "Hf177": 0.186,
        "Hf178": 0.2728,
        "Hf179": 0.1362,
        "Hf180": 0.3508,

    # Hg
        "Hg196": 0.0015,
        "Hg198": 0.0997,
        "Hg199": 0.1687,
        "Hg200": 0.231,
        "Hg201": 0.1318,
        "Hg202": 0.2986,
        "Hg204": 0.0687,

    # Ho
        "Ho165": 1,

    # I
        "I127": 1,

    # In
        "In113": 0.0429,
        "In115": 0.9571,

    # Ir
        "Ir191": 0.373,
        "Ir193": 0.627,

    # Kr
        "Kr78": 0.00355,
        "Kr80": 0.02286,
        "Kr82": 0.11593,
        "Kr83": 0.115,
        "Kr84": 0.56987,
        "Kr86": 0.17279,

    # La
        "La138": 0.0009,
        "La139": 0.9991,

    # Lu
        "Lu175": 0.9741,
        "Lu176": 0.0259,

    # Mg
        "Mg24": 0.7899,
        "Mg25": 0.1,
        "Mg26": 0.1101,

    # Mo
        "Mo92": 0.1484,
        "Mo94": 0.0925,
        "Mo95": 0.1592,
        "Mo96": 0.1668,
        "Mo97": 0.0955,
        "Mo98": 0.2413,
        "Mo100": 0.0963,

    # Mn
        "Mn55": 1,

    # Nd
        "Nd142": 0.2713,
        "Nd143": 0.1218,
        "Nd144": 0.23798,
        "Nd145": 0.08293,
        "Nd146": 0.17187,
        "Nd148": 0.05756,
        "Nd150": 0.05638,

    # Nb
        "Nb93": 1,

    # Ni
        "Ni58": 0.68077,
        "Ni60": 0.26223,
        "Ni61": 0.01139,
        "Ni62": 0.03634,
        "Ni64": 0.00926,

    # Os
        "Os184": 0.0002,
        "Os186": 0.0159,
        "Os187": 0.0196,
        "Os188": 0.1324,
        "Os189": 0.1615,
        "Os190": 0.2626,
        "Os192": 0.4078,

    # Pb
        "Pb204": 0.014,
        "Pb206": 0.241,
        "Pb207": 0.221,
        "Pb208": 0.524,

    # Pd
        "Pd102": 0.0102,
        "Pd104": 0.1114,
        "Pd105": 0.2233,
        "Pd106": 0.2733,
        "Pd108": 0.2646,
        "Pd110": 0.1172,

    # Pr
        "Pr141": 1,

    # Pt
        "Pt190": 0.0001,
        "Pt192": 0.0078,
        "Pt194": 0.3286,
        "Pt195": 0.3378,
        "Pt196": 0.2521,
        "Pt198": 0.0739,

    # Rb
        "Rb85": 0.7217,
        "Rb87": 0.2783,

    # Re
        "Re185": 0.374,
        "Re187": 0.626,

    # Rh
        "Rh103": 1,

    # Ru
        "Ru96": 0.0554,
        "Ru98": 0.0187,
        "Ru99": 0.1276,
        "Ru100": 0.126,
        "Ru101": 0.1706,
        "Ru102": 0.3155,
        "Ru104": 0.1862,

    # Sb
        "Sb121": 0.5721,
        "Sb123": 0.4279,

    # Sc
        "Sc45": 1,

    # Se
        "Se74": 0.0089,
        "Se76": 0.0937,
        "Se77": 0.0763,
        "Se78": 0.2377,
        "Se80": 0.4961,
        "Se82": 0.0873,

    # Si
        "Si28": 0.92223,
        "Si29": 0.04685,
        "Si30": 0.03092,

    # Sm
        "Sm144": 0.0307,
        "Sm147": 0.1499,
        "Sm148": 0.1124,
        "Sm149": 0.1382,
        "Sm150": 0.0738,
        "Sm152": 0.2675,
        "Sm154": 0.2275,

    # Sn
        "Sn112": 0.0097,
        "Sn114": 0.0066,
        "Sn115": 0.0034,
        "Sn116": 0.1454,
        "Sn117": 0.0768,
        "Sn118": 0.2422,
        "Sn119": 0.0859,
        "Sn120": 0.3258,
        "Sn122": 0.0463,
        "Sn124": 0.0579,

    # Sr
        "Sr84": 0.0056,
        "Sr86": 0.0986,
        "Sr87": 0.07,
        "Sr88": 0.8258,

    # Ta
        "Ta180": 0.00015,
        "Ta181": 0.99985,

    # Tb
        "Tb159": 1,

    # Te
        "Te120": 0.0009,
        "Te122": 0.0255,
        "Te123": 0.0089,
        "Te124": 0.0474,
        "Te125": 0.0707,
        "Te126": 0.1884,
        "Te128": 0.3174,
        "Te130": 0.3408,

    # Th
        "Th232": 1,

    # Ti
        "Ti46": 0.0825,
        "Ti47": 0.0744,
        "Ti48": 0.7372,
        "Ti49": 0.0541,
        "Ti50": 0.0518,

    # Tl
        "Tl203": 0.29524,
        "Tl205": 0.70476,

    # Tm
        "Tm169": 1,

    # U
        "U234": 5.4e-05,
        "U235": 0.007204,
        "U238": 0.992742,

    # V
        "V50": 0.0025,
        "V51": 0.9975,

    # W
        "W180": 0.0012,
        "W182": 0.265,
        "W183": 0.1431,
        "W184": 0.3064,
        "W186": 0.2845,

    # Xe
        "Xe124": 0.000952,
        "Xe126": 0.00089,
        "Xe128": 0.019102,
        "Xe129": 0.264006,
        "Xe130": 0.04071,
        "Xe131": 0.212324,
        "Xe132": 0.269086,
        "Xe134": 0.104357,
        "Xe136": 0.088573,

    # Y
        "Y89": 1,

    # Yb
        "Yb168": 0.0013,
        "Yb170": 0.0304,
        "Yb171": 0.1428,
        "Yb172": 0.2168,
        "Yb173": 0.161,
        "Yb174": 0.3203,
        "Yb176": 0.1272,

    # Zn
        "Zn64": 0.4917,
        "Zn66": 0.2773,
        "Zn67": 0.0404,
        "Zn68": 0.1845,
        "Zn70": 0.0061,

    # Zr
        "Zr90": 0.5145,
        "Zr91": 0.1122,
        "Zr92": 0.1715,
        "Zr94": 0.1738,
        "Zr96": 0.028,

    }


    clean_isotopes = {
        "Mg": ["Mg24", "Mg25", "Mg26"],
        "Ca": ["Ca40", "Ca42", "Ca43", "Ca44"],
        "Ti": ["Ti47", "Ti49"],
        "Cr": ["Cr52", "Cr53"],
        "Fe": ["Fe56", "Fe57"],
        "Ni": ["Ni60", "Ni61", "Ni62"],
        "Cu": ["Cu63", "Cu65"],
        "Zn": ["Zn66", "Zn67", "Zn68"],
        "Ga": ["Ga69", "Ga71"],
        "Ge": ["Ge72", "Ge73"],
        "Se": ["Se77", "Se78"],
        "Br": ["Br79", "Br81"],
        "Zr": ["Zr90", "Zr91"],
        "Mo": ["Mo95", "Mo97"],
        "Ru": ["Ru99", "Ru101"],
        "Pd": ["Pd105"],
        "Ag": ["Ag107", "Ag109"],
        "Cd": ["Cd111"],
        "Sn": ["Sn117", "Sn118", "Sn119"],
        "Sb": ["Sb121"],
        "Te": ["Te125", "Te126", "Te128"],
        "Ba": ["Ba135", "Ba137"],
        "Nd": ["Nd143", "Nd145", "Nd146"],
        "Sm": ["Sm147", "Sm148", "Sm149"],
        "Eu": ["Eu151", "Eu153"],
        "Gd": ["Gd155", "Gd157"],
        "Dy": ["Dy161", "Dy163"],
        "Er": ["Er166", "Er167"],
        "Yb": ["Yb171", "Yb172", "Yb173"],
        "Hf": ["Hf177", "Hf178", "Hf179"],
        "W": ["W182", "W183"],
        "Re": ["Re185"],
        "Os": ["Os188", "Os189"],
        "Ir": ["Ir191", "Ir193"],
        "Pt": ["Pt194", "Pt195"],
        "Tl": ["Tl203", "Tl205"],
        "Pb": ["Pb206", "Pb207", "Pb208"],
    }


    added_isotopes = {
        "Mg": ["Mg24", "Mg25", "Mg26"],
        "Ca": ["Ca40", "Ca42", "Ca43", "Ca44"],
        "Ti": ["Ti46", "Ti47", "Ti48", "Ti49"],
        "Cr": ["Cr52", "Cr53"],
        "Fe": ["Fe56", "Fe57"],
        "Ni": ["Ni58", "Ni60", "Ni61", "Ni62"],
        "Cu": ["Cu63", "Cu65"],
        "Zn": ["Zn64", "Zn66", "Zn67", "Zn68"],
        "Ga": ["Ga69", "Ga71"],
        "Ge": ["Ge70", "Ge72", "Ge73", "Ge74"],
        "Se": ["Se77", "Se78", "Se80"],
        "Br": ["Br79", "Br81"],
        "Zr": ["Zr90", "Zr91"],
        "Mo": ["Mo95", "Mo97"],
        "Ru": ["Ru99", "Ru101", "Ru102"],
        "Pd": ["Pd105", "Pd106", "Pd108"],
        "Ag": ["Ag107", "Ag109"],
        "Cd": ["Cd111", "Cd112", "Cd114"],
        "Sn": ["Sn117", "Sn118", "Sn119", "Sn120"],
        "Sb": ["Sb121", "Sb123"],
        "Te": ["Te125", "Te126", "Te128", "Te130"],
        "Ba": ["Ba135", "Ba137", "Ba138"],
        "Nd": ["Nd143", "Nd144", "Nd145", "Nd146"],
        "Sm": ["Sm147", "Sm148", "Sm149", "Sm152", "Sm154"],
        "Eu": ["Eu151", "Eu153"],
        "Gd": ["Gd155", "Gd156", "Gd157", "Gd158", "Gd160"],
        "Dy": ["Dy161", "Dy162", "Dy163", "Dy164"],
        "Er": ["Er166", "Er167", "Er168", "Er170"],
        "Yb": ["Yb171", "Yb172", "Yb173", "Yb174"],
        "Hf": ["Hf177", "Hf178", "Hf179", "Hf180"],
        "W": ["W182", "W183", "W184", "W186"],
        "Re": ["Re185", "Re187"],
        "Os": ["Os188", "Os189", "Os190", "Os192"],
        "Ir": ["Ir191", "Ir193"],
        "Pt": ["Pt194", "Pt195", "Pt196"],
        "Tl": ["Tl203", "Tl205"],
        "Pb": ["Pb206", "Pb207", "Pb208"],
    }


    isobaric_subtractions = {
        "Ti": ["Ca46", "Ca48"],
        "Ni": ["Fe58"],
        "Zn": ["Ni64"],
        "Ge": ["Zn70", "Se74"],
        "Se": ["Kr78", "Kr80"],
        "Ru": ["Pd102"],
        "Pd": ["Cd106", "Cd108"],
        "Cd": ["Sn112", "Sn114"],
        "Sn": ["Te120"],
        "Sb": ["Te123"],
        "Te": ["Xe130", "Ba130"],
        "Ba": ["La138", "Ce138"],
        "Nd": ["Sm144"],
        "Sm": ["Gd152", "Gd154"],
        "Gd": ["Dy156", "Dy158", "Dy160"],
        "Dy": ["Er162", "Er164"],
        "Er": ["Yb168", "Yb170"],
        "Yb": ["Hf174"],
        "Hf": ["Ta180", "W180"],
        "W": ["Os184", "Os186"],
        "Re": ["Os187"],
        "Os": ["Pt190", "Pt192"],
        "Pt": ["Hg196"],
    }


    beta_pairs = {
        "Ti": ["Ca43", "Ca44"],
        "Ni": ["Fe56", "Fe57"],
        "Zn": ["Ni61", "Ni62"],
        "Cd": ["Sn117", "Sn119"],
        "Te": ["Xe129", "Xe131", "Ba135", "Ba137"],
        "Nd": ["Sm147", "Sm149"],
        "Sm": ["Gd155", "Gd157"],
        "Gd": ["Dy161", "Dy163"],
        "Dy": ["Er166", "Er167"],
        "Er": ["Yb171", "Yb172"],
        "Yb": ["Hf177", "Hf178"],
        "W": ["Os188", "Os189"],
        "Re": ["Os188", "Os189"],
        "Os": ["Pt194", "Pt195"],
        "Pt": ["Hg199", "Hg200"],
    }

    ###############################################################################################


    ## calc starts here


    def _get_rm_name_for_spline():
        try:
            external_channels = [c for c in data.timeSeriesList(data.Input)
                                 if isinstance(c.property("External standard"), str) and c.property("External standard")]
            if external_channels:
                return external_channels[0].property("External standard").split(",")[0]
        except Exception:
            pass
        return None


    rmName_for_spmb = _get_rm_name_for_spline()


    def _copy_total_props(new_channel, anchor_ts):
        try:
            new_channel.setProperty("External standard", anchor_ts.property("External standard"))
            new_channel.setProperty("FitThroughZero", anchor_ts.property("FitThroughZero"))
            new_channel.setProperty("Model", anchor_ts.property("Model"))
            if bool(anchor_ts.property("BackgroundSubtracted")):
                new_channel.setProperty("BackgroundSubtracted", anchor_ts.property("BackgroundSubtracted"))
        except Exception as e:
            IoLog.warning(f"Could not copy channel properties: {e}")


    def _create_total_channel(channel_name, element, mass_number, values, anchor_ts):
        ch = data.createTimeSeries(
            channel_name, data.Input, indexChannel.time(), values,
            {**commonProps, 'Element': element, 'Mass': mass_number}
        )
        _copy_total_props(ch, anchor_ts)
        return ch


    def _create_corr_channel(channel_name, element, isotope_key, values):
        return data.createTimeSeries(
            channel_name, data.Intermediate, indexChannel.time(), values,
            {'Element': element, 'Mass': isotope_masses.get(isotope_key, int(''.join([c for c in isotope_key if c.isdigit()]))), **commonProps}
        )


    def _nomb_estimate(ref_ts, interferer_iso, ref_iso):
        ratio_true = natural_abundances[interferer_iso] / natural_abundances[ref_iso]
        return ref_ts.data() * ratio_true


    def _beta_from_pair(beta_iso_num, beta_iso_den):
        num_ts = data.timeSeries(beta_iso_num)
        den_ts = data.timeSeries(beta_iso_den)
        ratio_meas = num_ts.data() / den_ts.data()
        ratio_true = natural_abundances[beta_iso_num] / natural_abundances[beta_iso_den]
        m_num = isotope_masses[beta_iso_num]
        m_den = isotope_masses[beta_iso_den]
        with np.errstate(divide='ignore', invalid='ignore'):
            beta = np.log(ratio_meas / ratio_true) / np.log(m_num / m_den)
        beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)
        return beta


    def _spline_beta(beta_name, beta_values):
        beta_clean = np.nan_to_num(beta_values, nan=0.0, posinf=0.0, neginf=0.0)
        beta_clipped = np.clip(beta_clean, -2.0, 2.0)

        try:
            data.createTimeSeries(beta_name, data.Output, indexChannel.time(), beta_clipped)
        except Exception:
            pass

        if rmName_for_spmb is None:
            IoLog.warning(f"{beta_name}: no external standard found for spline; using clipped beta directly.")
            return beta_clipped

        try:
            spline_obj = data.spline(rmName_for_spmb, beta_name)
            if spline_obj is None:
                IoLog.warning(f"{beta_name}: spline was null; using clipped beta directly.")
                return beta_clipped
            return np.nan_to_num(spline_obj.data(), nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as e:
            IoLog.warning(f"{beta_name}: spline failed ({e}); using clipped beta directly.")
            return beta_clipped


    def _spmb_estimate(ref_ts, interferer_iso, ref_iso, beta_values):
        ratio_true = natural_abundances[interferer_iso] / natural_abundances[ref_iso]
        m_interferer = isotope_masses[interferer_iso]
        m_ref = isotope_masses[ref_iso]
        corrected_ratio = ratio_true * (m_interferer / m_ref) ** beta_values
        return ref_ts.data() * corrected_ratio


    # Mg
    try:
        Mg24_ts = data.timeSeries("Mg24")
        Mg25_ts = data.timeSeries("Mg25")
        Mg26_ts = data.timeSeries("Mg26")

        Mg_total = Mg24_ts.data() + Mg25_ts.data() + Mg26_ts.data()
        MgTotalChannel = _create_total_channel("MgTotal", "Mg", 24, Mg_total, Mg24_ts)

    except Exception as e:
        IoLog.error(f"Mg total correction failed: {e}")


    # Ca
    try:
        Ca40_ts = data.timeSeries("Ca40")
        Ca42_ts = data.timeSeries("Ca42")
        Ca43_ts = data.timeSeries("Ca43")
        Ca44_ts = data.timeSeries("Ca44")

        Ca_total = Ca40_ts.data() + Ca42_ts.data() + Ca43_ts.data() + Ca44_ts.data()
        CaTotalChannel = _create_total_channel("CaTotal", "Ca", 40, Ca_total, Ca40_ts)

    except Exception as e:
        IoLog.error(f"Ca total correction failed: {e}")


    # Ti
    try:
        Ti47_ts = data.timeSeries("Ti47")
        Ti49_ts = data.timeSeries("Ti49")

        Ti_total = Ti47_ts.data() + Ti49_ts.data()
        TiTotalChannel = _create_total_channel("TiTotal", "Ti", 47, Ti_total, Ti47_ts)

    except Exception as e:
        IoLog.error(f"Ti total correction failed: {e}")

    try:
        Ti46_ts = data.timeSeries("Ti46")
        Ti47_ts = data.timeSeries("Ti47")
        Ti48_ts = data.timeSeries("Ti48")
        Ti49_ts = data.timeSeries("Ti49")
        Ca44_ts = data.timeSeries("Ca44")

        Ca46_estimate_nomb = _nomb_estimate(Ca44_ts, "Ca46", "Ca44")
        Ca48_estimate_nomb = _nomb_estimate(Ca44_ts, "Ca48", "Ca44")
        Ti46_corrected_nomb = Ti46_ts.data() - Ca46_estimate_nomb
        Ti48_corrected_nomb = Ti48_ts.data() - Ca48_estimate_nomb

        _create_corr_channel("Ti46_corrnomb", "Ti", "Ti46", Ti46_corrected_nomb)
        _create_corr_channel("Ti48_corrnomb", "Ti", "Ti48", Ti48_corrected_nomb)

        Ti_total_nomb = Ti47_ts.data() + Ti49_ts.data() + Ti46_corrected_nomb + Ti48_corrected_nomb
        TiTotal_nombChannel = _create_total_channel("TiTotal_nomb", "Ti", 47, Ti_total_nomb, Ti47_ts)

    except Exception as e:
        IoLog.error(f"Ti total nomb correction failed: {e}")

    try:
        Ti46_ts = data.timeSeries("Ti46")
        Ti47_ts = data.timeSeries("Ti47")
        Ti48_ts = data.timeSeries("Ti48")
        Ti49_ts = data.timeSeries("Ti49")
        Ca44_ts = data.timeSeries("Ca44")

        Ti48_ratio = natural_abundances["Ti48"] / natural_abundances["Ti49"]
        Ti48_estimate_nomb2 = Ti49_ts.data() * Ti48_ratio
        
        Ti48_corrected_nomb2 = Ti48_ts.data() - (Ti48_ts.data() - Ti48_estimate_nomb2)

        _create_corr_channel("Ti48_corrnomb2", "Ti", "Ti48", Ti48_corrected_nomb2)

        Ti_total_nomb2 = Ti49_ts.data() + Ti48_corrected_nomb2
        TiTotal_nomb2Channel = _create_total_channel("TiTotal_nomb2", "Ti", 48, Ti_total_nomb2, Ti48_ts)

    except Exception as e:
        IoLog.error(f"Ti total nomb correction failed: {e}")

    try:
        Ti46_ts = data.timeSeries("Ti46")
        Ti47_ts = data.timeSeries("Ti47")
        Ti48_ts = data.timeSeries("Ti48")
        Ti49_ts = data.timeSeries("Ti49")
        Ca44_ts = data.timeSeries("Ca44")

        Ti48_ratio = natural_abundances["Ti48"] / natural_abundances["Ti49"]
        Ti48_estimate_2 = Ti49_ts.data() * Ti48_ratio
        
        Ti48_corrected_2 = Ti48_ts.data() - (Ti48_ts.data() - Ti48_estimate_nomb2)

        _create_corr_channel("Ti48_corr2", "Ti", "Ti48", Ti48_corrected_2)

        Ti_48_2 = Ti48_corrected_2
        TiTotal_nomb2Channel = _create_total_channel("Ti_48_2", "Ti", 48, Ti_48_2, Ti48_ts)

    except Exception as e:
        IoLog.error(f"Ti total nomb correction failed: {e}")


    try:
        Ti46_ts = data.timeSeries("Ti46")
        Ti47_ts = data.timeSeries("Ti47")
        Ti48_ts = data.timeSeries("Ti48")
        Ti49_ts = data.timeSeries("Ti49")
        Ca44_ts = data.timeSeries("Ca44")

        CaBeta = _beta_from_pair("Ca43", "Ca44")
        CaBetaSpline = _spline_beta("CaBeta_for_Ti_spmb", CaBeta)

        Ca46_estimate_spmb = _spmb_estimate(Ca44_ts, "Ca46", "Ca44", CaBetaSpline)
        Ca48_estimate_spmb = _spmb_estimate(Ca44_ts, "Ca48", "Ca44", CaBetaSpline)
        Ti46_corrected_spmb = Ti46_ts.data() - Ca46_estimate_spmb
        Ti48_corrected_spmb = Ti48_ts.data() - Ca48_estimate_spmb

        _create_corr_channel("Ti46_corrspmb", "Ti", "Ti46", Ti46_corrected_spmb)
        _create_corr_channel("Ti48_corrspmb", "Ti", "Ti48", Ti48_corrected_spmb)

        Ti_total_spmb = Ti47_ts.data() + Ti49_ts.data() + Ti46_corrected_spmb + Ti48_corrected_spmb
        TiTotal_spmbChannel = _create_total_channel("TiTotal_spmb", "Ti", 47, Ti_total_spmb, Ti47_ts)

    except Exception as e:
        IoLog.error(f"Ti total spmb correction failed: {e}")


    # Cr
    try:
        Cr52_ts = data.timeSeries("Cr52")
        Cr53_ts = data.timeSeries("Cr53")

        Cr_total = Cr52_ts.data() + Cr53_ts.data()
        CrTotalChannel = _create_total_channel("CrTotal", "Cr", 52, Cr_total, Cr52_ts)

    except Exception as e:
        IoLog.error(f"Cr total correction failed: {e}")


    # Fe
    try:
        Fe56_ts = data.timeSeries("Fe56")
        Fe57_ts = data.timeSeries("Fe57")

        Fe_total = Fe56_ts.data() + Fe57_ts.data()
        FeTotalChannel = _create_total_channel("FeTotal", "Fe", 56, Fe_total, Fe56_ts)

    except Exception as e:
        IoLog.error(f"Fe total correction failed: {e}")


    # Ni
    try:
        Ni60_ts = data.timeSeries("Ni60")
        Ni61_ts = data.timeSeries("Ni61")
        Ni62_ts = data.timeSeries("Ni62")

        Ni_total = Ni60_ts.data() + Ni61_ts.data() + Ni62_ts.data()
        NiTotalChannel = _create_total_channel("NiTotal", "Ni", 60, Ni_total, Ni60_ts)

    except Exception as e:
        IoLog.error(f"Ni total correction failed: {e}")

    try:
        Ni58_ts = data.timeSeries("Ni58")
        Ni60_ts = data.timeSeries("Ni60")
        Ni61_ts = data.timeSeries("Ni61")
        Ni62_ts = data.timeSeries("Ni62")
        Fe57_ts = data.timeSeries("Fe57")

        Fe58_estimate_nomb = _nomb_estimate(Fe57_ts, "Fe58", "Fe57")
        Ni58_corrected_nomb = Ni58_ts.data() - Fe58_estimate_nomb
        _create_corr_channel("Ni58_corrnomb", "Ni", "Ni58", Ni58_corrected_nomb)

        Ni_total_nomb = Ni60_ts.data() + Ni61_ts.data() + Ni62_ts.data() + Ni58_corrected_nomb
        NiTotal_nombChannel = _create_total_channel("NiTotal_nomb", "Ni", 60, Ni_total_nomb, Ni60_ts)

    except Exception as e:
        IoLog.error(f"Ni total nomb correction failed: {e}")

    try:
        Ni58_ts = data.timeSeries("Ni58")
        Ni60_ts = data.timeSeries("Ni60")
        Ni61_ts = data.timeSeries("Ni61")
        Ni62_ts = data.timeSeries("Ni62")
        Fe57_ts = data.timeSeries("Fe57")

        FeBeta = _beta_from_pair("Fe56", "Fe57")
        FeBetaSpline = _spline_beta("FeBeta_for_Ni_spmb", FeBeta)

        Fe58_estimate_spmb = _spmb_estimate(Fe57_ts, "Fe58", "Fe57", FeBetaSpline)
        Ni58_corrected_spmb = Ni58_ts.data() - Fe58_estimate_spmb
        _create_corr_channel("Ni58_corrspmb", "Ni", "Ni58", Ni58_corrected_spmb)

        Ni_total_spmb = Ni60_ts.data() + Ni61_ts.data() + Ni62_ts.data() + Ni58_corrected_spmb
        NiTotal_spmbChannel = _create_total_channel("NiTotal_spmb", "Ni", 60, Ni_total_spmb, Ni60_ts)

    except Exception as e:
        IoLog.error(f"Ni total spmb correction failed: {e}")


    # Cu
    try:
        Cu63_ts = data.timeSeries("Cu63")
        Cu65_ts = data.timeSeries("Cu65")

        Cu_total = Cu63_ts.data() + Cu65_ts.data()
        CuTotalChannel = _create_total_channel("CuTotal", "Cu", 63, Cu_total, Cu63_ts)

    except Exception as e:
        IoLog.error(f"Cu total correction failed: {e}")


    # Zn
    try:
        Zn66_ts = data.timeSeries("Zn66")
        Zn67_ts = data.timeSeries("Zn67")
        Zn68_ts = data.timeSeries("Zn68")

        Zn_total = Zn66_ts.data() + Zn67_ts.data() + Zn68_ts.data()
        ZnTotalChannel = _create_total_channel("ZnTotal", "Zn", 66, Zn_total, Zn66_ts)

    except Exception as e:
        IoLog.error(f"Zn total correction failed: {e}")

    try:
        Zn64_ts = data.timeSeries("Zn64")
        Zn66_ts = data.timeSeries("Zn66")
        Zn67_ts = data.timeSeries("Zn67")
        Zn68_ts = data.timeSeries("Zn68")
        Ni62_ts = data.timeSeries("Ni62")

        Ni64_estimate_nomb = _nomb_estimate(Ni62_ts, "Ni64", "Ni62")
        Zn64_corrected_nomb = Zn64_ts.data() - Ni64_estimate_nomb
        _create_corr_channel("Zn64_corrnomb", "Zn", "Zn64", Zn64_corrected_nomb)

        Zn_total_nomb = Zn66_ts.data() + Zn67_ts.data() + Zn68_ts.data() + Zn64_corrected_nomb
        ZnTotal_nombChannel = _create_total_channel("ZnTotal_nomb", "Zn", 66, Zn_total_nomb, Zn66_ts)

    except Exception as e:
        IoLog.error(f"Zn total nomb correction failed: {e}")

    try:
        Zn64_ts = data.timeSeries("Zn64")
        Zn66_ts = data.timeSeries("Zn66")
        Zn67_ts = data.timeSeries("Zn67")
        Zn68_ts = data.timeSeries("Zn68")
        Ni62_ts = data.timeSeries("Ni62")

        NiBeta = _beta_from_pair("Ni61", "Ni62")
        NiBetaSpline = _spline_beta("NiBeta_for_Zn_spmb", NiBeta)

        Ni64_estimate_spmb = _spmb_estimate(Ni62_ts, "Ni64", "Ni62", NiBetaSpline)
        Zn64_corrected_spmb = Zn64_ts.data() - Ni64_estimate_spmb
        _create_corr_channel("Zn64_corrspmb", "Zn", "Zn64", Zn64_corrected_spmb)

        Zn_total_spmb = Zn66_ts.data() + Zn67_ts.data() + Zn68_ts.data() + Zn64_corrected_spmb
        ZnTotal_spmbChannel = _create_total_channel("ZnTotal_spmb", "Zn", 66, Zn_total_spmb, Zn66_ts)

    except Exception as e:
        IoLog.error(f"Zn total spmb correction failed: {e}")


    # Ga
    try:
        Ga69_ts = data.timeSeries("Ga69")
        Ga71_ts = data.timeSeries("Ga71")

        Ga_total = Ga69_ts.data() + Ga71_ts.data()
        GaTotalChannel = _create_total_channel("GaTotal", "Ga", 69, Ga_total, Ga69_ts)

    except Exception as e:
        IoLog.error(f"Ga total correction failed: {e}")


    # Ge
    try:
        Ge72_ts = data.timeSeries("Ge72")
        Ge73_ts = data.timeSeries("Ge73")

        Ge_total = Ge72_ts.data() + Ge73_ts.data()
        GeTotalChannel = _create_total_channel("GeTotal", "Ge", 72, Ge_total, Ge72_ts)

    except Exception as e:
        IoLog.error(f"Ge total correction failed: {e}")

    try:
        Ge70_ts = data.timeSeries("Ge70")
        Ge72_ts = data.timeSeries("Ge72")
        Ge73_ts = data.timeSeries("Ge73")
        Ge74_ts = data.timeSeries("Ge74")
        Zn68_ts = data.timeSeries("Zn68")
        Se77_ts = data.timeSeries("Se77")

        Zn70_estimate_nomb = _nomb_estimate(Zn68_ts, "Zn70", "Zn68")
        Se74_estimate_nomb = _nomb_estimate(Se77_ts, "Se74", "Se77")
        Ge70_corrected_nomb = Ge70_ts.data() - Zn70_estimate_nomb
        Ge74_corrected_nomb = Ge74_ts.data() - Se74_estimate_nomb

        _create_corr_channel("Ge70_corrnomb", "Ge", "Ge70", Ge70_corrected_nomb)
        _create_corr_channel("Ge74_corrnomb", "Ge", "Ge74", Ge74_corrected_nomb)

        Ge_total_nomb = Ge72_ts.data() + Ge73_ts.data() + Ge70_corrected_nomb + Ge74_corrected_nomb
        GeTotal_nombChannel = _create_total_channel("GeTotal_nomb", "Ge", 72, Ge_total_nomb, Ge72_ts)

    except Exception as e:
        IoLog.error(f"Ge total nomb correction failed: {e}")


    # Se
    try:
        Se77_ts = data.timeSeries("Se77")

        Se_total = Se77_ts.data()
        SeTotalChannel = _create_total_channel("SeTotal", "Se", 77, Se_total, Se77_ts)

    except Exception as e:
        IoLog.error(f"Se total correction failed: {e}")

    try:
        Se77_ts = data.timeSeries("Se77")
        Se78_ts = data.timeSeries("Se78")
        Se80_ts = data.timeSeries("Se80")
        Kr83_ts = data.timeSeries("Kr83")

        Kr78_estimate_nomb = _nomb_estimate(Kr83_ts, "Kr78", "Kr83")
        Kr80_estimate_nomb = _nomb_estimate(Kr83_ts, "Kr80", "Kr83")
        Se78_corrected_nomb = Se78_ts.data() - Kr78_estimate_nomb
        Se80_corrected_nomb = Se80_ts.data() - Kr80_estimate_nomb

        _create_corr_channel("Se78_corrnomb", "Se", "Se78", Se78_corrected_nomb)
        _create_corr_channel("Se80_corrnomb", "Se", "Se80", Se80_corrected_nomb)

        Se_total_nomb = Se77_ts.data() + Se78_corrected_nomb + Se80_corrected_nomb
        SeTotal_nombChannel = _create_total_channel("SeTotal_nomb", "Se", 77, Se_total_nomb, Se77_ts)

    except Exception as e:
        IoLog.error(f"Se total nomb correction failed: {e}")


    # Br
    try:
        Br79_ts = data.timeSeries("Br79")
        Br81_ts = data.timeSeries("Br81")

        Br_total = Br79_ts.data() + Br81_ts.data()
        BrTotalChannel = _create_total_channel("BrTotal", "Br", 79, Br_total, Br79_ts)

    except Exception as e:
        IoLog.error(f"Br total correction failed: {e}")


    # Zr
    try:
        Zr90_ts = data.timeSeries("Zr90")
        Zr91_ts = data.timeSeries("Zr91")

        Zr_total = Zr90_ts.data() + Zr91_ts.data()
        ZrTotalChannel = _create_total_channel("ZrTotal", "Zr", 90, Zr_total, Zr90_ts)

    except Exception as e:
        IoLog.error(f"Zr total correction failed: {e}")


    # Mo
    try:
        Mo95_ts = data.timeSeries("Mo95")
        Mo97_ts = data.timeSeries("Mo97")

        Mo_total = Mo95_ts.data() + Mo97_ts.data()
        MoTotalChannel = _create_total_channel("MoTotal", "Mo", 95, Mo_total, Mo95_ts)

    except Exception as e:
        IoLog.error(f"Mo total correction failed: {e}")


    # Ru
    try:
        Ru99_ts = data.timeSeries("Ru99")
        Ru101_ts = data.timeSeries("Ru101")

        Ru_total = Ru99_ts.data() + Ru101_ts.data()
        RuTotalChannel = _create_total_channel("RuTotal", "Ru", 101, Ru_total, Ru101_ts)

    except Exception as e:
        IoLog.error(f"Ru total correction failed: {e}")

    try:
        Ru99_ts = data.timeSeries("Ru99")
        Ru101_ts = data.timeSeries("Ru101")
        Ru102_ts = data.timeSeries("Ru102")
        Pd105_ts = data.timeSeries("Pd105")

        Pd102_estimate_nomb = _nomb_estimate(Pd105_ts, "Pd102", "Pd105")
        Ru102_corrected_nomb = Ru102_ts.data() - Pd102_estimate_nomb
        _create_corr_channel("Ru102_corrnomb", "Ru", "Ru102", Ru102_corrected_nomb)

        Ru_total_nomb = Ru99_ts.data() + Ru101_ts.data() + Ru102_corrected_nomb
        RuTotal_nombChannel = _create_total_channel("RuTotal_nomb", "Ru", 101, Ru_total_nomb, Ru101_ts)

    except Exception as e:
        IoLog.error(f"Ru total nomb correction failed: {e}")


    # Pd
    try:
        Pd105_ts = data.timeSeries("Pd105")

        Pd_total = Pd105_ts.data()
        PdTotalChannel = _create_total_channel("PdTotal", "Pd", 105, Pd_total, Pd105_ts)

    except Exception as e:
        IoLog.error(f"Pd total correction failed: {e}")

    try:
        Pd105_ts = data.timeSeries("Pd105")
        Pd106_ts = data.timeSeries("Pd106")
        Pd108_ts = data.timeSeries("Pd108")
        Cd111_ts = data.timeSeries("Cd111")

        Cd106_estimate_nomb = _nomb_estimate(Cd111_ts, "Cd106", "Cd111")
        Cd108_estimate_nomb = _nomb_estimate(Cd111_ts, "Cd108", "Cd111")
        Pd106_corrected_nomb = Pd106_ts.data() - Cd106_estimate_nomb
        Pd108_corrected_nomb = Pd108_ts.data() - Cd108_estimate_nomb

        _create_corr_channel("Pd106_corrnomb", "Pd", "Pd106", Pd106_corrected_nomb)
        _create_corr_channel("Pd108_corrnomb", "Pd", "Pd108", Pd108_corrected_nomb)

        Pd_total_nomb = Pd105_ts.data() + Pd106_corrected_nomb + Pd108_corrected_nomb
        PdTotal_nombChannel = _create_total_channel("PdTotal_nomb", "Pd", 105, Pd_total_nomb, Pd105_ts)

    except Exception as e:
        IoLog.error(f"Pd total nomb correction failed: {e}")


    # Ag
    try:
        Ag107_ts = data.timeSeries("Ag107")
        Ag109_ts = data.timeSeries("Ag109")

        Ag_total = Ag107_ts.data() + Ag109_ts.data()
        AgTotalChannel = _create_total_channel("AgTotal", "Ag", 107, Ag_total, Ag107_ts)

    except Exception as e:
        IoLog.error(f"Ag total correction failed: {e}")


    # Cd
    try:
        Cd111_ts = data.timeSeries("Cd111")

        Cd_total = Cd111_ts.data()
        CdTotalChannel = _create_total_channel("CdTotal", "Cd", 111, Cd_total, Cd111_ts)

    except Exception as e:
        IoLog.error(f"Cd total correction failed: {e}")

    try:
        Cd111_ts = data.timeSeries("Cd111")
        Cd112_ts = data.timeSeries("Cd112")
        Cd114_ts = data.timeSeries("Cd114")
        Sn119_ts = data.timeSeries("Sn119")

        Sn112_estimate_nomb = _nomb_estimate(Sn119_ts, "Sn112", "Sn119")
        Sn114_estimate_nomb = _nomb_estimate(Sn119_ts, "Sn114", "Sn119")
        Cd112_corrected_nomb = Cd112_ts.data() - Sn112_estimate_nomb
        Cd114_corrected_nomb = Cd114_ts.data() - Sn114_estimate_nomb

        _create_corr_channel("Cd112_corrnomb", "Cd", "Cd112", Cd112_corrected_nomb)
        _create_corr_channel("Cd114_corrnomb", "Cd", "Cd114", Cd114_corrected_nomb)

        Cd_total_nomb = Cd111_ts.data() + Cd112_corrected_nomb + Cd114_corrected_nomb
        CdTotal_nombChannel = _create_total_channel("CdTotal_nomb", "Cd", 111, Cd_total_nomb, Cd111_ts)

    except Exception as e:
        IoLog.error(f"Cd total nomb correction failed: {e}")

    try:
        Cd111_ts = data.timeSeries("Cd111")
        Cd112_ts = data.timeSeries("Cd112")
        Cd114_ts = data.timeSeries("Cd114")
        Sn119_ts = data.timeSeries("Sn119")

        SnBeta = _beta_from_pair("Sn117", "Sn119")
        SnBetaSpline = _spline_beta("SnBeta_for_Cd_spmb", SnBeta)

        Sn112_estimate_spmb = _spmb_estimate(Sn119_ts, "Sn112", "Sn119", SnBetaSpline)
        Sn114_estimate_spmb = _spmb_estimate(Sn119_ts, "Sn114", "Sn119", SnBetaSpline)
        Cd112_corrected_spmb = Cd112_ts.data() - Sn112_estimate_spmb
        Cd114_corrected_spmb = Cd114_ts.data() - Sn114_estimate_spmb

        _create_corr_channel("Cd112_corrspmb", "Cd", "Cd112", Cd112_corrected_spmb)
        _create_corr_channel("Cd114_corrspmb", "Cd", "Cd114", Cd114_corrected_spmb)

        Cd_total_spmb = Cd111_ts.data() + Cd112_corrected_spmb + Cd114_corrected_spmb
        CdTotal_spmbChannel = _create_total_channel("CdTotal_spmb", "Cd", 111, Cd_total_spmb, Cd111_ts)

    except Exception as e:
        IoLog.error(f"Cd total spmb correction failed: {e}")


    # Sn
    try:
        Sn117_ts = data.timeSeries("Sn117")
        Sn118_ts = data.timeSeries("Sn118")
        Sn119_ts = data.timeSeries("Sn119")

        Sn_total = Sn117_ts.data() + Sn118_ts.data() + Sn119_ts.data()
        SnTotalChannel = _create_total_channel("SnTotal", "Sn", 118, Sn_total, Sn118_ts)

    except Exception as e:
        IoLog.error(f"Sn total correction failed: {e}")

    try:
        Sn117_ts = data.timeSeries("Sn117")
        Sn118_ts = data.timeSeries("Sn118")
        Sn119_ts = data.timeSeries("Sn119")
        Sn120_ts = data.timeSeries("Sn120")
        Te125_ts = data.timeSeries("Te125")

        Te120_estimate_nomb = _nomb_estimate(Te125_ts, "Te120", "Te125")
        Sn120_corrected_nomb = Sn120_ts.data() - Te120_estimate_nomb
        _create_corr_channel("Sn120_corrnomb", "Sn", "Sn120", Sn120_corrected_nomb)

        Sn_total_nomb = Sn117_ts.data() + Sn118_ts.data() + Sn119_ts.data() + Sn120_corrected_nomb
        SnTotal_nombChannel = _create_total_channel("SnTotal_nomb", "Sn", 118, Sn_total_nomb, Sn118_ts)

    except Exception as e:
        IoLog.error(f"Sn total nomb correction failed: {e}")


    # Sb
    try:
        Sb121_ts = data.timeSeries("Sb121")

        Sb_total = Sb121_ts.data()
        SbTotalChannel = _create_total_channel("SbTotal", "Sb", 121, Sb_total, Sb121_ts)

    except Exception as e:
        IoLog.error(f"Sb total correction failed: {e}")

    try:
        Sb121_ts = data.timeSeries("Sb121")
        Sb123_ts = data.timeSeries("Sb123")
        Te125_ts = data.timeSeries("Te125")

        Te123_estimate_nomb = _nomb_estimate(Te125_ts, "Te123", "Te125")
        Sb123_corrected_nomb = Sb123_ts.data() - Te123_estimate_nomb
        _create_corr_channel("Sb123_corrnomb", "Sb", "Sb123", Sb123_corrected_nomb)

        Sb_total_nomb = Sb121_ts.data() + Sb123_corrected_nomb
        SbTotal_nombChannel = _create_total_channel("SbTotal_nomb", "Sb", 121, Sb_total_nomb, Sb121_ts)

    except Exception as e:
        IoLog.error(f"Sb total nomb correction failed: {e}")


    # Te
    try:
        Te125_ts = data.timeSeries("Te125")
        Te126_ts = data.timeSeries("Te126")
        Te128_ts = data.timeSeries("Te128")

        Te_total = Te125_ts.data() + Te126_ts.data() + Te128_ts.data()
        TeTotalChannel = _create_total_channel("TeTotal", "Te", 125, Te_total, Te125_ts)

    except Exception as e:
        IoLog.error(f"Te total correction failed: {e}")

    try:
        Te125_ts = data.timeSeries("Te125")
        Te126_ts = data.timeSeries("Te126")
        Te128_ts = data.timeSeries("Te128")
        Te130_ts = data.timeSeries("Te130")
        Xe131_ts = data.timeSeries("Xe131")
        Ba137_ts = data.timeSeries("Ba137")

        Xe130_estimate_nomb = _nomb_estimate(Xe131_ts, "Xe130", "Xe131")
        Ba130_estimate_nomb = _nomb_estimate(Ba137_ts, "Ba130", "Ba137")
        Te130_corrected_nomb = Te130_ts.data() - Xe130_estimate_nomb - Ba130_estimate_nomb
        _create_corr_channel("Te130_corrnomb", "Te", "Te130", Te130_corrected_nomb)

        Te_total_nomb = Te125_ts.data() + Te126_ts.data() + Te128_ts.data() + Te130_corrected_nomb
        TeTotal_nombChannel = _create_total_channel("TeTotal_nomb", "Te", 125, Te_total_nomb, Te125_ts)

    except Exception as e:
        IoLog.error(f"Te total nomb correction failed: {e}")

    try:
        Te125_ts = data.timeSeries("Te125")
        Te126_ts = data.timeSeries("Te126")
        Te128_ts = data.timeSeries("Te128")
        Te130_ts = data.timeSeries("Te130")
        Xe131_ts = data.timeSeries("Xe131")
        Ba137_ts = data.timeSeries("Ba137")

        XeBeta = _beta_from_pair("Xe129", "Xe131")
        XeBetaSpline = _spline_beta("XeBeta_for_Te_spmb", XeBeta)
        BaBeta = _beta_from_pair("Ba135", "Ba137")
        BaBetaSpline = _spline_beta("BaBeta_for_Te_spmb", BaBeta)

        Xe130_estimate_spmb = _spmb_estimate(Xe131_ts, "Xe130", "Xe131", XeBetaSpline)
        Ba130_estimate_spmb = _spmb_estimate(Ba137_ts, "Ba130", "Ba137", BaBetaSpline)
        Te130_corrected_spmb = Te130_ts.data() - Xe130_estimate_spmb - Ba130_estimate_spmb
        _create_corr_channel("Te130_corrspmb", "Te", "Te130", Te130_corrected_spmb)

        Te_total_spmb = Te125_ts.data() + Te126_ts.data() + Te128_ts.data() + Te130_corrected_spmb
        TeTotal_spmbChannel = _create_total_channel("TeTotal_spmb", "Te", 125, Te_total_spmb, Te125_ts)

    except Exception as e:
        IoLog.error(f"Te total spmb correction failed: {e}")


    # Ba
    try:
        Ba135_ts = data.timeSeries("Ba135")
        Ba137_ts = data.timeSeries("Ba137")

        Ba_total = Ba135_ts.data() + Ba137_ts.data()
        BaTotalChannel = _create_total_channel("BaTotal", "Ba", 137, Ba_total, Ba137_ts)

    except Exception as e:
        IoLog.error(f"Ba total correction failed: {e}")

    try:
        Ba135_ts = data.timeSeries("Ba135")
        Ba137_ts = data.timeSeries("Ba137")
        Ba138_ts = data.timeSeries("Ba138")
        La139_ts = data.timeSeries("La139")
        Ce140_ts = data.timeSeries("Ce140")

        La138_estimate_nomb = _nomb_estimate(La139_ts, "La138", "La139")
        Ce138_estimate_nomb = _nomb_estimate(Ce140_ts, "Ce138", "Ce140")
        Ba138_corrected_nomb = Ba138_ts.data() - La138_estimate_nomb - Ce138_estimate_nomb
        _create_corr_channel("Ba138_corrnomb", "Ba", "Ba138", Ba138_corrected_nomb)

        Ba_total_nomb = Ba135_ts.data() + Ba137_ts.data() + Ba138_corrected_nomb
        BaTotal_nombChannel = _create_total_channel("BaTotal_nomb", "Ba", 137, Ba_total_nomb, Ba137_ts)

    except Exception as e:
        IoLog.error(f"Ba total nomb correction failed: {e}")


    # Nd
    try:
        Nd143_ts = data.timeSeries("Nd143")
        Nd145_ts = data.timeSeries("Nd145")
        Nd146_ts = data.timeSeries("Nd146")

        Nd_total = Nd143_ts.data() + Nd145_ts.data() + Nd146_ts.data()
        NdTotalChannel = _create_total_channel("NdTotal", "Nd", 146, Nd_total, Nd146_ts)

    except Exception as e:
        IoLog.error(f"Nd total correction failed: {e}")

    try:
        Nd143_ts = data.timeSeries("Nd143")
        Nd144_ts = data.timeSeries("Nd144")
        Nd145_ts = data.timeSeries("Nd145")
        Nd146_ts = data.timeSeries("Nd146")
        Sm149_ts = data.timeSeries("Sm149")

        Sm144_estimate_nomb = _nomb_estimate(Sm149_ts, "Sm144", "Sm149")
        Nd144_corrected_nomb = Nd144_ts.data() - Sm144_estimate_nomb
        _create_corr_channel("Nd144_corrnomb", "Nd", "Nd144", Nd144_corrected_nomb)

        Nd_total_nomb = Nd143_ts.data() + Nd145_ts.data() + Nd146_ts.data() + Nd144_corrected_nomb
        NdTotal_nombChannel = _create_total_channel("NdTotal_nomb", "Nd", 146, Nd_total_nomb, Nd146_ts)

    except Exception as e:
        IoLog.error(f"Nd total nomb correction failed: {e}")

    try:
        Nd143_ts = data.timeSeries("Nd143")
        Nd144_ts = data.timeSeries("Nd144")
        Nd145_ts = data.timeSeries("Nd145")
        Nd146_ts = data.timeSeries("Nd146")
        Sm149_ts = data.timeSeries("Sm149")

        SmBeta = _beta_from_pair("Sm147", "Sm149")
        SmBetaSpline = _spline_beta("SmBeta_for_Nd_spmb", SmBeta)

        Sm144_estimate_spmb = _spmb_estimate(Sm149_ts, "Sm144", "Sm149", SmBetaSpline)
        Nd144_corrected_spmb = Nd144_ts.data() - Sm144_estimate_spmb
        _create_corr_channel("Nd144_corrspmb", "Nd", "Nd144", Nd144_corrected_spmb)

        Nd_total_spmb = Nd143_ts.data() + Nd145_ts.data() + Nd146_ts.data() + Nd144_corrected_spmb
        NdTotal_spmbChannel = _create_total_channel("NdTotal_spmb", "Nd", 146, Nd_total_spmb, Nd146_ts)

    except Exception as e:
        IoLog.error(f"Nd total spmb correction failed: {e}")


    # Sm
    try:
        Sm147_ts = data.timeSeries("Sm147")
        Sm148_ts = data.timeSeries("Sm148")
        Sm149_ts = data.timeSeries("Sm149")

        Sm_total = Sm147_ts.data() + Sm148_ts.data() + Sm149_ts.data()
        SmTotalChannel = _create_total_channel("SmTotal", "Sm", 147, Sm_total, Sm147_ts)

    except Exception as e:
        IoLog.error(f"Sm total correction failed: {e}")

    try:
        Sm147_ts = data.timeSeries("Sm147")
        Sm148_ts = data.timeSeries("Sm148")
        Sm149_ts = data.timeSeries("Sm149")
        Sm152_ts = data.timeSeries("Sm152")
        Sm154_ts = data.timeSeries("Sm154")
        Gd157_ts = data.timeSeries("Gd157")

        Gd152_estimate_nomb = _nomb_estimate(Gd157_ts, "Gd152", "Gd157")
        Gd154_estimate_nomb = _nomb_estimate(Gd157_ts, "Gd154", "Gd157")
        Sm152_corrected_nomb = Sm152_ts.data() - Gd152_estimate_nomb
        Sm154_corrected_nomb = Sm154_ts.data() - Gd154_estimate_nomb

        _create_corr_channel("Sm152_corrnomb", "Sm", "Sm152", Sm152_corrected_nomb)
        _create_corr_channel("Sm154_corrnomb", "Sm", "Sm154", Sm154_corrected_nomb)

        Sm_total_nomb = Sm147_ts.data() + Sm148_ts.data() + Sm149_ts.data() + Sm152_corrected_nomb + Sm154_corrected_nomb
        SmTotal_nombChannel = _create_total_channel("SmTotal_nomb", "Sm", 147, Sm_total_nomb, Sm147_ts)

    except Exception as e:
        IoLog.error(f"Sm total nomb correction failed: {e}")

    try:
        Sm147_ts = data.timeSeries("Sm147")
        Sm148_ts = data.timeSeries("Sm148")
        Sm149_ts = data.timeSeries("Sm149")
        Sm152_ts = data.timeSeries("Sm152")
        Sm154_ts = data.timeSeries("Sm154")
        Gd157_ts = data.timeSeries("Gd157")

        GdBeta = _beta_from_pair("Gd155", "Gd157")
        GdBetaSpline = _spline_beta("GdBeta_for_Sm_spmb", GdBeta)

        Gd152_estimate_spmb = _spmb_estimate(Gd157_ts, "Gd152", "Gd157", GdBetaSpline)
        Gd154_estimate_spmb = _spmb_estimate(Gd157_ts, "Gd154", "Gd157", GdBetaSpline)
        Sm152_corrected_spmb = Sm152_ts.data() - Gd152_estimate_spmb
        Sm154_corrected_spmb = Sm154_ts.data() - Gd154_estimate_spmb

        _create_corr_channel("Sm152_corrspmb", "Sm", "Sm152", Sm152_corrected_spmb)
        _create_corr_channel("Sm154_corrspmb", "Sm", "Sm154", Sm154_corrected_spmb)

        Sm_total_spmb = Sm147_ts.data() + Sm148_ts.data() + Sm149_ts.data() + Sm152_corrected_spmb + Sm154_corrected_spmb
        SmTotal_spmbChannel = _create_total_channel("SmTotal_spmb", "Sm", 147, Sm_total_spmb, Sm147_ts)

    except Exception as e:
        IoLog.error(f"Sm total spmb correction failed: {e}")


    # Eu
    try:
        Eu151_ts = data.timeSeries("Eu151")
        Eu153_ts = data.timeSeries("Eu153")

        Eu_total = Eu151_ts.data() + Eu153_ts.data()
        EuTotalChannel = _create_total_channel("EuTotal", "Eu", 153, Eu_total, Eu153_ts)

    except Exception as e:
        IoLog.error(f"Eu total correction failed: {e}")


    # Gd
    try:
        Gd155_ts = data.timeSeries("Gd155")
        Gd157_ts = data.timeSeries("Gd157")

        Gd_total = Gd155_ts.data() + Gd157_ts.data()
        GdTotalChannel = _create_total_channel("GdTotal", "Gd", 157, Gd_total, Gd157_ts)

    except Exception as e:
        IoLog.error(f"Gd total correction failed: {e}")

    try:
        Gd155_ts = data.timeSeries("Gd155")
        Gd156_ts = data.timeSeries("Gd156")
        Gd157_ts = data.timeSeries("Gd157")
        Gd158_ts = data.timeSeries("Gd158")
        Gd160_ts = data.timeSeries("Gd160")
        Dy163_ts = data.timeSeries("Dy163")

        Dy156_estimate_nomb = _nomb_estimate(Dy163_ts, "Dy156", "Dy163")
        Dy158_estimate_nomb = _nomb_estimate(Dy163_ts, "Dy158", "Dy163")
        Dy160_estimate_nomb = _nomb_estimate(Dy163_ts, "Dy160", "Dy163")
        Gd156_corrected_nomb = Gd156_ts.data() - Dy156_estimate_nomb
        Gd158_corrected_nomb = Gd158_ts.data() - Dy158_estimate_nomb
        Gd160_corrected_nomb = Gd160_ts.data() - Dy160_estimate_nomb

        _create_corr_channel("Gd156_corrnomb", "Gd", "Gd156", Gd156_corrected_nomb)
        _create_corr_channel("Gd158_corrnomb", "Gd", "Gd158", Gd158_corrected_nomb)
        _create_corr_channel("Gd160_corrnomb", "Gd", "Gd160", Gd160_corrected_nomb)

        Gd_total_nomb = Gd155_ts.data() + Gd157_ts.data() + Gd156_corrected_nomb + Gd158_corrected_nomb + Gd160_corrected_nomb
        GdTotal_nombChannel = _create_total_channel("GdTotal_nomb", "Gd", 157, Gd_total_nomb, Gd157_ts)

    except Exception as e:
        IoLog.error(f"Gd total nomb correction failed: {e}")

    try:
        Gd155_ts = data.timeSeries("Gd155")
        Gd156_ts = data.timeSeries("Gd156")
        Gd157_ts = data.timeSeries("Gd157")
        Gd158_ts = data.timeSeries("Gd158")
        Gd160_ts = data.timeSeries("Gd160")
        Dy163_ts = data.timeSeries("Dy163")

        DyBeta = _beta_from_pair("Dy161", "Dy163")
        DyBetaSpline = _spline_beta("DyBeta_for_Gd_spmb", DyBeta)

        Dy156_estimate_spmb = _spmb_estimate(Dy163_ts, "Dy156", "Dy163", DyBetaSpline)
        Dy158_estimate_spmb = _spmb_estimate(Dy163_ts, "Dy158", "Dy163", DyBetaSpline)
        Dy160_estimate_spmb = _spmb_estimate(Dy163_ts, "Dy160", "Dy163", DyBetaSpline)
        Gd156_corrected_spmb = Gd156_ts.data() - Dy156_estimate_spmb
        Gd158_corrected_spmb = Gd158_ts.data() - Dy158_estimate_spmb
        Gd160_corrected_spmb = Gd160_ts.data() - Dy160_estimate_spmb

        _create_corr_channel("Gd156_corrspmb", "Gd", "Gd156", Gd156_corrected_spmb)
        _create_corr_channel("Gd158_corrspmb", "Gd", "Gd158", Gd158_corrected_spmb)
        _create_corr_channel("Gd160_corrspmb", "Gd", "Gd160", Gd160_corrected_spmb)

        Gd_total_spmb = Gd155_ts.data() + Gd157_ts.data() + Gd156_corrected_spmb + Gd158_corrected_spmb + Gd160_corrected_spmb
        GdTotal_spmbChannel = _create_total_channel("GdTotal_spmb", "Gd", 157, Gd_total_spmb, Gd157_ts)

    except Exception as e:
        IoLog.error(f"Gd total spmb correction failed: {e}")


    # Dy
    try:
        Dy161_ts = data.timeSeries("Dy161")
        Dy163_ts = data.timeSeries("Dy163")

        Dy_total = Dy161_ts.data() + Dy163_ts.data()
        DyTotalChannel = _create_total_channel("DyTotal", "Dy", 163, Dy_total, Dy163_ts)

    except Exception as e:
        IoLog.error(f"Dy total correction failed: {e}")

    try:
        Dy161_ts = data.timeSeries("Dy161")
        Dy162_ts = data.timeSeries("Dy162")
        Dy163_ts = data.timeSeries("Dy163")
        Dy164_ts = data.timeSeries("Dy164")
        Er167_ts = data.timeSeries("Er167")

        Er162_estimate_nomb = _nomb_estimate(Er167_ts, "Er162", "Er167")
        Er164_estimate_nomb = _nomb_estimate(Er167_ts, "Er164", "Er167")
        Dy162_corrected_nomb = Dy162_ts.data() - Er162_estimate_nomb
        Dy164_corrected_nomb = Dy164_ts.data() - Er164_estimate_nomb

        _create_corr_channel("Dy162_corrnomb", "Dy", "Dy162", Dy162_corrected_nomb)
        _create_corr_channel("Dy164_corrnomb", "Dy", "Dy164", Dy164_corrected_nomb)

        Dy_total_nomb = Dy161_ts.data() + Dy163_ts.data() + Dy162_corrected_nomb + Dy164_corrected_nomb
        DyTotal_nombChannel = _create_total_channel("DyTotal_nomb", "Dy", 163, Dy_total_nomb, Dy163_ts)

    except Exception as e:
        IoLog.error(f"Dy total nomb correction failed: {e}")

    try:
        Dy161_ts = data.timeSeries("Dy161")
        Dy162_ts = data.timeSeries("Dy162")
        Dy163_ts = data.timeSeries("Dy163")
        Dy164_ts = data.timeSeries("Dy164")
        Er167_ts = data.timeSeries("Er167")

        ErBeta = _beta_from_pair("Er166", "Er167")
        ErBetaSpline = _spline_beta("ErBeta_for_Dy_spmb", ErBeta)

        Er162_estimate_spmb = _spmb_estimate(Er167_ts, "Er162", "Er167", ErBetaSpline)
        Er164_estimate_spmb = _spmb_estimate(Er167_ts, "Er164", "Er167", ErBetaSpline)
        Dy162_corrected_spmb = Dy162_ts.data() - Er162_estimate_spmb
        Dy164_corrected_spmb = Dy164_ts.data() - Er164_estimate_spmb

        _create_corr_channel("Dy162_corrspmb", "Dy", "Dy162", Dy162_corrected_spmb)
        _create_corr_channel("Dy164_corrspmb", "Dy", "Dy164", Dy164_corrected_spmb)

        Dy_total_spmb = Dy161_ts.data() + Dy163_ts.data() + Dy162_corrected_spmb + Dy164_corrected_spmb
        DyTotal_spmbChannel = _create_total_channel("DyTotal_spmb", "Dy", 163, Dy_total_spmb, Dy163_ts)

    except Exception as e:
        IoLog.error(f"Dy total spmb correction failed: {e}")


    # Er
    try:
        Er166_ts = data.timeSeries("Er166")
        Er167_ts = data.timeSeries("Er167")

        Er_total = Er166_ts.data() + Er167_ts.data()
        ErTotalChannel = _create_total_channel("ErTotal", "Er", 166, Er_total, Er166_ts)

    except Exception as e:
        IoLog.error(f"Er total correction failed: {e}")

    try:
        Er166_ts = data.timeSeries("Er166")
        Er167_ts = data.timeSeries("Er167")
        Er168_ts = data.timeSeries("Er168")
        Er170_ts = data.timeSeries("Er170")
        Yb172_ts = data.timeSeries("Yb172")

        Yb168_estimate_nomb = _nomb_estimate(Yb172_ts, "Yb168", "Yb172")
        Yb170_estimate_nomb = _nomb_estimate(Yb172_ts, "Yb170", "Yb172")
        Er168_corrected_nomb = Er168_ts.data() - Yb168_estimate_nomb
        Er170_corrected_nomb = Er170_ts.data() - Yb170_estimate_nomb

        _create_corr_channel("Er168_corrnomb", "Er", "Er168", Er168_corrected_nomb)
        _create_corr_channel("Er170_corrnomb", "Er", "Er170", Er170_corrected_nomb)

        Er_total_nomb = Er166_ts.data() + Er167_ts.data() + Er168_corrected_nomb + Er170_corrected_nomb
        ErTotal_nombChannel = _create_total_channel("ErTotal_nomb", "Er", 166, Er_total_nomb, Er166_ts)

    except Exception as e:
        IoLog.error(f"Er total nomb correction failed: {e}")

    try:
        Er166_ts = data.timeSeries("Er166")
        Er167_ts = data.timeSeries("Er167")
        Er168_ts = data.timeSeries("Er168")
        Er170_ts = data.timeSeries("Er170")
        Yb172_ts = data.timeSeries("Yb172")

        YbBeta = _beta_from_pair("Yb171", "Yb172")
        YbBetaSpline = _spline_beta("YbBeta_for_Er_spmb", YbBeta)

        Yb168_estimate_spmb = _spmb_estimate(Yb172_ts, "Yb168", "Yb172", YbBetaSpline)
        Yb170_estimate_spmb = _spmb_estimate(Yb172_ts, "Yb170", "Yb172", YbBetaSpline)
        Er168_corrected_spmb = Er168_ts.data() - Yb168_estimate_spmb
        Er170_corrected_spmb = Er170_ts.data() - Yb170_estimate_spmb

        _create_corr_channel("Er168_corrspmb", "Er", "Er168", Er168_corrected_spmb)
        _create_corr_channel("Er170_corrspmb", "Er", "Er170", Er170_corrected_spmb)

        Er_total_spmb = Er166_ts.data() + Er167_ts.data() + Er168_corrected_spmb + Er170_corrected_spmb
        ErTotal_spmbChannel = _create_total_channel("ErTotal_spmb", "Er", 166, Er_total_spmb, Er166_ts)

    except Exception as e:
        IoLog.error(f"Er total spmb correction failed: {e}")


    # Yb
    try:
        Yb171_ts = data.timeSeries("Yb171")
        Yb172_ts = data.timeSeries("Yb172")
        Yb173_ts = data.timeSeries("Yb173")

        Yb_total = Yb171_ts.data() + Yb172_ts.data() + Yb173_ts.data()
        YbTotalChannel = _create_total_channel("YbTotal", "Yb", 172, Yb_total, Yb172_ts)

    except Exception as e:
        IoLog.error(f"Yb total correction failed: {e}")

    try:
        Yb171_ts = data.timeSeries("Yb171")
        Yb172_ts = data.timeSeries("Yb172")
        Yb173_ts = data.timeSeries("Yb173")
        Yb174_ts = data.timeSeries("Yb174")
        Hf178_ts = data.timeSeries("Hf178")

        Hf174_estimate_nomb = _nomb_estimate(Hf178_ts, "Hf174", "Hf178")
        Yb174_corrected_nomb = Yb174_ts.data() - Hf174_estimate_nomb
        _create_corr_channel("Yb174_corrnomb", "Yb", "Yb174", Yb174_corrected_nomb)

        Yb_total_nomb = Yb171_ts.data() + Yb172_ts.data() + Yb173_ts.data() + Yb174_corrected_nomb
        YbTotal_nombChannel = _create_total_channel("YbTotal_nomb", "Yb", 172, Yb_total_nomb, Yb172_ts)

    except Exception as e:
        IoLog.error(f"Yb total nomb correction failed: {e}")

    try:
        Yb171_ts = data.timeSeries("Yb171")
        Yb172_ts = data.timeSeries("Yb172")
        Yb173_ts = data.timeSeries("Yb173")
        Yb174_ts = data.timeSeries("Yb174")
        Hf178_ts = data.timeSeries("Hf178")

        HfBeta = _beta_from_pair("Hf177", "Hf178")
        HfBetaSpline = _spline_beta("HfBeta_for_Yb_spmb", HfBeta)

        Hf174_estimate_spmb = _spmb_estimate(Hf178_ts, "Hf174", "Hf178", HfBetaSpline)
        Yb174_corrected_spmb = Yb174_ts.data() - Hf174_estimate_spmb
        _create_corr_channel("Yb174_corrspmb", "Yb", "Yb174", Yb174_corrected_spmb)

        Yb_total_spmb = Yb171_ts.data() + Yb172_ts.data() + Yb173_ts.data() + Yb174_corrected_spmb
        YbTotal_spmbChannel = _create_total_channel("YbTotal_spmb", "Yb", 172, Yb_total_spmb, Yb172_ts)

    except Exception as e:
        IoLog.error(f"Yb total spmb correction failed: {e}")


    # Hf
    try:
        Hf177_ts = data.timeSeries("Hf177")
        Hf178_ts = data.timeSeries("Hf178")
        Hf179_ts = data.timeSeries("Hf179")

        Hf_total = Hf177_ts.data() + Hf178_ts.data() + Hf179_ts.data()
        HfTotalChannel = _create_total_channel("HfTotal", "Hf", 178, Hf_total, Hf178_ts)

    except Exception as e:
        IoLog.error(f"Hf total correction failed: {e}")

    try:
        Hf177_ts = data.timeSeries("Hf177")
        Hf178_ts = data.timeSeries("Hf178")
        Hf179_ts = data.timeSeries("Hf179")
        Hf180_ts = data.timeSeries("Hf180")
        Ta181_ts = data.timeSeries("Ta181")
        W182_ts = data.timeSeries("W182")

        Ta180_estimate_nomb = _nomb_estimate(Ta181_ts, "Ta180", "Ta181")
        W180_estimate_nomb = _nomb_estimate(W182_ts, "W180", "W182")
        Hf180_corrected_nomb = Hf180_ts.data() - Ta180_estimate_nomb - W180_estimate_nomb
        _create_corr_channel("Hf180_corrnomb", "Hf", "Hf180", Hf180_corrected_nomb)

        Hf_total_nomb = Hf177_ts.data() + Hf178_ts.data() + Hf179_ts.data() + Hf180_corrected_nomb
        HfTotal_nombChannel = _create_total_channel("HfTotal_nomb", "Hf", 178, Hf_total_nomb, Hf178_ts)

    except Exception as e:
        IoLog.error(f"Hf total nomb correction failed: {e}")


    # W
    try:
        W182_ts = data.timeSeries("W182")
        W183_ts = data.timeSeries("W183")

        W_total = W182_ts.data() + W183_ts.data()
        WTotalChannel = _create_total_channel("WTotal", "W", 182, W_total, W182_ts)

    except Exception as e:
        IoLog.error(f"W total correction failed: {e}")

    try:
        W182_ts = data.timeSeries("W182")
        W183_ts = data.timeSeries("W183")
        W184_ts = data.timeSeries("W184")
        W186_ts = data.timeSeries("W186")
        Os189_ts = data.timeSeries("Os189")

        Os184_estimate_nomb = _nomb_estimate(Os189_ts, "Os184", "Os189")
        Os186_estimate_nomb = _nomb_estimate(Os189_ts, "Os186", "Os189")
        W184_corrected_nomb = W184_ts.data() - Os184_estimate_nomb
        W186_corrected_nomb = W186_ts.data() - Os186_estimate_nomb

        _create_corr_channel("W184_corrnomb", "W", "W184", W184_corrected_nomb)
        _create_corr_channel("W186_corrnomb", "W", "W186", W186_corrected_nomb)

        W_total_nomb = W182_ts.data() + W183_ts.data() + W184_corrected_nomb + W186_corrected_nomb
        WTotal_nombChannel = _create_total_channel("WTotal_nomb", "W", 182, W_total_nomb, W182_ts)

    except Exception as e:
        IoLog.error(f"W total nomb correction failed: {e}")

    try:
        W182_ts = data.timeSeries("W182")
        W183_ts = data.timeSeries("W183")
        W184_ts = data.timeSeries("W184")
        W186_ts = data.timeSeries("W186")
        Os189_ts = data.timeSeries("Os189")

        OsBeta = _beta_from_pair("Os188", "Os189")
        OsBetaSpline = _spline_beta("OsBeta_for_W_spmb", OsBeta)

        Os184_estimate_spmb = _spmb_estimate(Os189_ts, "Os184", "Os189", OsBetaSpline)
        Os186_estimate_spmb = _spmb_estimate(Os189_ts, "Os186", "Os189", OsBetaSpline)
        W184_corrected_spmb = W184_ts.data() - Os184_estimate_spmb
        W186_corrected_spmb = W186_ts.data() - Os186_estimate_spmb

        _create_corr_channel("W184_corrspmb", "W", "W184", W184_corrected_spmb)
        _create_corr_channel("W186_corrspmb", "W", "W186", W186_corrected_spmb)

        W_total_spmb = W182_ts.data() + W183_ts.data() + W184_corrected_spmb + W186_corrected_spmb
        WTotal_spmbChannel = _create_total_channel("WTotal_spmb", "W", 182, W_total_spmb, W182_ts)

    except Exception as e:
        IoLog.error(f"W total spmb correction failed: {e}")


    # Re
    try:
        Re185_ts = data.timeSeries("Re185")

        Re_total = Re185_ts.data()
        ReTotalChannel = _create_total_channel("ReTotal", "Re", 185, Re_total, Re185_ts)

    except Exception as e:
        IoLog.error(f"Re total correction failed: {e}")

    try:
        Re185_ts = data.timeSeries("Re185")
        Re187_ts = data.timeSeries("Re187")
        Os189_ts = data.timeSeries("Os189")

        Os187_estimate_nomb = _nomb_estimate(Os189_ts, "Os187", "Os189")
        Re187_corrected_nomb = Re187_ts.data() - Os187_estimate_nomb
        _create_corr_channel("Re187_corrnomb", "Re", "Re187", Re187_corrected_nomb)

        Re_total_nomb = Re185_ts.data() + Re187_corrected_nomb
        ReTotal_nombChannel = _create_total_channel("ReTotal_nomb", "Re", 185, Re_total_nomb, Re185_ts)

    except Exception as e:
        IoLog.error(f"Re total nomb correction failed: {e}")

    try:
        Re185_ts = data.timeSeries("Re185")
        Re187_ts = data.timeSeries("Re187")
        Os189_ts = data.timeSeries("Os189")

        OsBeta = _beta_from_pair("Os188", "Os189")
        OsBetaSpline = _spline_beta("OsBeta_for_Re_spmb", OsBeta)

        Os187_estimate_spmb = _spmb_estimate(Os189_ts, "Os187", "Os189", OsBetaSpline)
        Re187_corrected_spmb = Re187_ts.data() - Os187_estimate_spmb
        _create_corr_channel("Re187_corrspmb", "Re", "Re187", Re187_corrected_spmb)

        Re_total_spmb = Re185_ts.data() + Re187_corrected_spmb
        ReTotal_spmbChannel = _create_total_channel("ReTotal_spmb", "Re", 185, Re_total_spmb, Re185_ts)

    except Exception as e:
        IoLog.error(f"Re total spmb correction failed: {e}")


    # Os
    try:
        Os188_ts = data.timeSeries("Os188")
        Os189_ts = data.timeSeries("Os189")

        Os_total = Os188_ts.data() + Os189_ts.data()
        OsTotalChannel = _create_total_channel("OsTotal", "Os", 189, Os_total, Os189_ts)

    except Exception as e:
        IoLog.error(f"Os total correction failed: {e}")

    try:
        Os188_ts = data.timeSeries("Os188")
        Os189_ts = data.timeSeries("Os189")
        Os190_ts = data.timeSeries("Os190")
        Os192_ts = data.timeSeries("Os192")
        Pt195_ts = data.timeSeries("Pt195")

        Pt190_estimate_nomb = _nomb_estimate(Pt195_ts, "Pt190", "Pt195")
        Pt192_estimate_nomb = _nomb_estimate(Pt195_ts, "Pt192", "Pt195")
        Os190_corrected_nomb = Os190_ts.data() - Pt190_estimate_nomb
        Os192_corrected_nomb = Os192_ts.data() - Pt192_estimate_nomb

        _create_corr_channel("Os190_corrnomb", "Os", "Os190", Os190_corrected_nomb)
        _create_corr_channel("Os192_corrnomb", "Os", "Os192", Os192_corrected_nomb)

        Os_total_nomb = Os188_ts.data() + Os189_ts.data() + Os190_corrected_nomb + Os192_corrected_nomb
        OsTotal_nombChannel = _create_total_channel("OsTotal_nomb", "Os", 189, Os_total_nomb, Os189_ts)

    except Exception as e:
        IoLog.error(f"Os total nomb correction failed: {e}")

    try:
        Os188_ts = data.timeSeries("Os188")
        Os189_ts = data.timeSeries("Os189")
        Os190_ts = data.timeSeries("Os190")
        Os192_ts = data.timeSeries("Os192")
        Pt195_ts = data.timeSeries("Pt195")

        PtBeta = _beta_from_pair("Pt194", "Pt195")
        PtBetaSpline = _spline_beta("PtBeta_for_Os_spmb", PtBeta)

        Pt190_estimate_spmb = _spmb_estimate(Pt195_ts, "Pt190", "Pt195", PtBetaSpline)
        Pt192_estimate_spmb = _spmb_estimate(Pt195_ts, "Pt192", "Pt195", PtBetaSpline)
        Os190_corrected_spmb = Os190_ts.data() - Pt190_estimate_spmb
        Os192_corrected_spmb = Os192_ts.data() - Pt192_estimate_spmb

        _create_corr_channel("Os190_corrspmb", "Os", "Os190", Os190_corrected_spmb)
        _create_corr_channel("Os192_corrspmb", "Os", "Os192", Os192_corrected_spmb)

        Os_total_spmb = Os188_ts.data() + Os189_ts.data() + Os190_corrected_spmb + Os192_corrected_spmb
        OsTotal_spmbChannel = _create_total_channel("OsTotal_spmb", "Os", 189, Os_total_spmb, Os189_ts)

    except Exception as e:
        IoLog.error(f"Os total spmb correction failed: {e}")


    # Ir
    try:
        Ir191_ts = data.timeSeries("Ir191")
        Ir193_ts = data.timeSeries("Ir193")

        Ir_total = Ir191_ts.data() + Ir193_ts.data()
        IrTotalChannel = _create_total_channel("IrTotal", "Ir", 193, Ir_total, Ir193_ts)

    except Exception as e:
        IoLog.error(f"Ir total correction failed: {e}")


    # Pt
    try:
        Pt194_ts = data.timeSeries("Pt194")
        Pt195_ts = data.timeSeries("Pt195")

        Pt_total = Pt194_ts.data() + Pt195_ts.data()
        PtTotalChannel = _create_total_channel("PtTotal", "Pt", 195, Pt_total, Pt195_ts)

    except Exception as e:
        IoLog.error(f"Pt total correction failed: {e}")

    try:
        Pt194_ts = data.timeSeries("Pt194")
        Pt195_ts = data.timeSeries("Pt195")
        Pt196_ts = data.timeSeries("Pt196")
        Hg200_ts = data.timeSeries("Hg200")

        Hg196_estimate_nomb = _nomb_estimate(Hg200_ts, "Hg196", "Hg200")
        Pt196_corrected_nomb = Pt196_ts.data() - Hg196_estimate_nomb
        _create_corr_channel("Pt196_corrnomb", "Pt", "Pt196", Pt196_corrected_nomb)

        Pt_total_nomb = Pt194_ts.data() + Pt195_ts.data() + Pt196_corrected_nomb
        PtTotal_nombChannel = _create_total_channel("PtTotal_nomb", "Pt", 195, Pt_total_nomb, Pt195_ts)

    except Exception as e:
        IoLog.error(f"Pt total nomb correction failed: {e}")

    try:
        Pt194_ts = data.timeSeries("Pt194")
        Pt195_ts = data.timeSeries("Pt195")
        Pt196_ts = data.timeSeries("Pt196")
        Hg200_ts = data.timeSeries("Hg200")

        HgBeta = _beta_from_pair("Hg199", "Hg200")
        HgBetaSpline = _spline_beta("HgBeta_for_Pt_spmb", HgBeta)

        Hg196_estimate_spmb = _spmb_estimate(Hg200_ts, "Hg196", "Hg200", HgBetaSpline)
        Pt196_corrected_spmb = Pt196_ts.data() - Hg196_estimate_spmb
        _create_corr_channel("Pt196_corrspmb", "Pt", "Pt196", Pt196_corrected_spmb)

        Pt_total_spmb = Pt194_ts.data() + Pt195_ts.data() + Pt196_corrected_spmb
        PtTotal_spmbChannel = _create_total_channel("PtTotal_spmb", "Pt", 195, Pt_total_spmb, Pt195_ts)

    except Exception as e:
        IoLog.error(f"Pt total spmb correction failed: {e}")


    # Tl
    try:
        Tl203_ts = data.timeSeries("Tl203")
        Tl205_ts = data.timeSeries("Tl205")

        Tl_total = Tl203_ts.data() + Tl205_ts.data()
        TlTotalChannel = _create_total_channel("TlTotal", "Tl", 205, Tl_total, Tl205_ts)

    except Exception as e:
        IoLog.error(f"Tl total correction failed: {e}")


    # Pb
    try:
        Pb206_ts = data.timeSeries("Pb206")
        Pb207_ts = data.timeSeries("Pb207")
        Pb208_ts = data.timeSeries("Pb208")

        Pb_total = Pb206_ts.data() + Pb207_ts.data() + Pb208_ts.data()
        PbTotalChannel = _create_total_channel("PbTotal", "Pb", 208, Pb_total, Pb208_ts)

    except Exception as e:
        IoLog.error(f"Pb total correction failed: {e}")


    ####################################################################

    # Baseline Subtraction
    if bl_required:
        drs.baselineSubtract(blGrp, data.timeSeriesList(data.Input), mask, 10, 20)
    else:
        drs.baselineSubtract(None, data.timeSeriesList(data.Input), mask, 10, 20)


    # Check that some Externals have been selected:
    atLeastOneExt = False
    for channel in data.timeSeriesList(data.Input):
        if type(channel.property("External standard")) == str:
            atLeastOneExt = True
            break

    if not atLeastOneExt:
        IoLog.error(f"No external standards are set. Please set at least one RM before continuing.")
        drs.message.emit("DRS did not finish.")
        drs.progress.emit(100)
        drs.finished.emit()
        return

    # Find blocks
    drs.message.emit('Finding blocks')
    drs.progress.emit(23)
    cal = Calibration()
    # the user may have selected an external that doesn't exist. If so, it will raise a RuntimeError.
    try:
        cal.updateBlocks()
    except MissingRMGroupError:
        # Find which RM is missing:
        externalsInUse = set(list(itertools.chain(*[c.property('External standard').split(',') for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name and type(c.property('External standard')) == str])))
        if "Model" in externalsInUse:
            externalsInUse.remove("Model")
        for ext in externalsInUse:
            try:
                data.selectionGroup(ext)
            except RuntimeError:
                IoLog.error(f"External standard {ext} is set but no selections are made for it.")
                drs.message.emit("DRS did not finish.")
                drs.progress.emit(100)
                drs.finished.emit()
                return

    # Calculate SQ channels
    for ii, input in enumerate(data.timeSeriesList(data.Input)):
        if 'TotalBeam' in input.name:
            continue

        if input.property('External standard') == '':
            print(f'No external standard for {input.name}')
            continue

        drs.progress.emit(25 + 25*float(ii)/len(data.timeSeriesList(data.Input)))
        drs.message.emit(f'Applying surface for {input.name}')

        try:
            surface = cal.surface(input.name, inv=True)
        except:
            IoLog.warning(
                f"There was an issue calculating the calibration surface for {input.name}. \
                This may be due to missing concentration values for your reference materials.")
            continue

        if not surface:
            IoLog.warning(f'No surface for {input.name}')
            continue

        cps = data.timeSeries(f'{input.name}_CPS')
        ppm = surface(cps.time(), cps.data())

        props = { **commonProps,
            'Element': input.property('Element'),
            'Mass': input.property('Mass'),
            'Reference Material': input.property('External standard'),
            'Reference Material uncertainty': cal.uncertainty(input.name), # This needs some thinking....
            'Units': 'µg.g-1',
            'AssociatedInputChannel': input.name
        }

        data.createTimeSeries(f'{input.name}_ppm', data.Output, indexChannel.time(), ppm, props)

    sels = list(itertools.chain(*[sg.selections() for sg in data.selectionGroupList(data.ReferenceMaterial | data.Sample)]))

    mfc = np.ones(len(indexChannel.time()))

    externalsInUse = assignExternalAffinities()
    affIndex = data.createTimeSeriesFromMetadata('ExtAffinityIndex', 'External affinity')

    # Optionally apply a fractionation correction

    if np.any([c.property('FractionationCorrection') for c in data.timeSeriesList(data.Input)]) and settings['UseIntStds']:
        print('Attempting fractionation correction...')
        allSels = [[s for s in sg.selections()] for sg in data.selectionGroupList(data.ReferenceMaterial | data.Sample)]
        allSels = list(itertools.chain.from_iterable(allSels))
        allIS = [sel.property('Internal element') for sel in allSels]
        isElementsList = list(set(allIS))

        if None in isElementsList:
            QMessageBox.warning(None, "Warning", "Some selections do not have an internal standard element set. Fractionation correction was not applied...")
        # Should be good to do fractionation correction now...
        else:
            isElementsList.sort()

            params = {}
            for c in [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]:
                params[c.name] = cal.fractionation(c.name)

            bs = data.timeSeries('BeamSeconds').data()
            isIndex = data.createTimeSeriesFromMetadata('ISElementIndex', 'Internal element')

            for c in [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]:
                fdf = params[c.name]
                if len(fdf) == 0:
                    print(f'No frac for {c.name}')
                    continue

                for isi, ise in enumerate(isElementsList):
                    for gi, g in enumerate(externalsInUse):
                        t, r, rsd, sp = cal.fitFractionation(c.name, ise, group=g)
                        if not sp:
                            continue
                        ind = np.where( (isIndex.data() == isi) & (affIndex.data() == gi))[0]
                        ppm = data.timeSeries(f'{c.name}_ppm')
                        ppmd = ppm.data()
                        ppmd[ind] = ppmd[ind]/(sp(bs[ind]))
                        ppm.setData(ppmd)

    if False:
        # This is a bit of code for experimenting with forcing certain elements
        # in unknown samples to be homogeneous prior to normalization.
        # This can flatten out any remaining DH fractionation in
        # unknowns, but there may still be fractionation for other reasons
        for ppmc in data.timeSeriesList(data.Output, {'Units': 'µg.g-1'}):
            if ppmc.name not in ['Al27_ppm', 'Ca43_ppm', 'Fe57_ppm', 'Mg25_ppm']:
                continue
            for si, sel in enumerate(sels):
                selIS = data.timeSeries(sel.property('Internal element')+'_ppm').dataForSelection(sel)
                selIS = savgol_filter(selIS, 21, 3)
                selC = ppmc.dataForSelection(sel)
                selC = savgol_filter(selC, 21, 3)
                r = (selC/np.nanmedian(selC))/(selIS/np.nanmedian(selIS))
                ppmcd = ppmc.data()
                ppmcd[ppmc.selectionIndices(sel)] /= r
                ppmc.setData(ppmcd)

    # If using fg RM data, need to divide by (thickness * area^2 * density)
    if drs.setting('UseFG'):
        massChannel = data.createTimeSeries('AblatedMass', data.Intermediate, indexChannel.time(), np.ones(len(indexChannel.time())))
        # TODO: Use laser log to auto get spot size
        for group in data.selectionGroupList(data.Sample | data.ReferenceMaterial):
            try:
                ind = list(itertools.chain(*[indexChannel.selectionIndices(s) for s in group.selections()]))
                density = float(group.property('Density'))*1e21 # Units of fg/m^3 here, g/cc in dialog
                thickness = float(group.property('Thickness'))*1e-6 # Units of m here, um in dialog
                spot_size = float(group.property('SpotSize'))*1e-6 # Units of m here, um in dialog
                spot_is_rect = group.property('SpotShape') == 'rect'
                if spot_is_rect:
                    spot_area = spot_size*spot_size
                else:
                    spot_area = PI * (spot_size/2)**2

                massChannel.data()[ind] = density*spot_area*thickness
            except Exception as e:
                print(f'Exception while calculating ablated mass per shot for group {group.name}: {e}')

        for c in data.timeSeriesList(data.Output):
            if not c.name.endswith('ppm'):
                continue

            c.setData(1e6 * c.data() / massChannel.data())



    # Calculate FQ
    if bool(settings['UseIntStds']):
        if not np.any([bool(sel.property('Internal value')) for sel in sels]) and not np.any([sel.property('Internal element') == 'Criteria' for sel in sels]):
            IoLog.error("No internal standard values found. Have you set any?")
            drs.message.emit("DRS did not finish.")
            drs.progress.emit(100)
            drs.finished.emit()
            return

        data.createTimeSeriesFromMetadata('ISValue', 'Internal value', sels)

        norm = np.ones(len(indexChannel.time()))
        crit_index = np.empty(len(indexChannel.time()))
        crit_index[:] = np.nan

        criteria_globals = {'where': np.where}
        for c in data.timeSeriesList(data.Input):
            try:
                criteria_globals[c.name] = data.timeSeries(f'{c.name}_CPS').data()
            except:
                pass

        try:
            criteria_list = settings['ISCriteria']
            for i in range(len(criteria_list)):
                criteria_list[i]['criteria'] = eval(f"lambda: where({criteria_list[i]['criteria']})", criteria_globals)
                criteria_list[i]['indicies'] = criteria_list[i]['criteria']()
        except Exception as e:
            pass
            #print(f'Not using criteria or there was a problem parsing the criteria...')

        ppmd = lambda s: data.timeSeries(f'{s}_ppm').data()

        for ii, sel in enumerate(sels):
            isvalue = sel.property('Internal value')
            iselement = sel.property('Internal element')
            isunit = sel.property('Internal units')
            si = indexChannel.selectionIndices(sel)

            if ii%20 == 0:
                drs.progress.emit(50 + 40*float(ii)/len(sels))
                drs.message.emit('Applying internal standards')

            if iselement == 'Criteria':
                for ci, criteria in enumerate(criteria_list):
                    csi = np.intersect1d(si, criteria['indicies'])
                    if len(csi) == 0:
                        continue
                    sum = np.zeros(len(csi))

                    for analyte in criteria['analytes'].split(','):
                        if not analyte:
                            continue
                        f = 1
                        if criteria['oxides']:
                            el = data.timeSeries(analyte).property('Element')
                            if el in ','.join(criteria['oxide_forms']):
                                matches = [f for f in criteria['oxide_forms'] if el in f]
                                if matches:
                                    f = data.oxideToElementFactor(matches[0])
                            else:
                                f = data.oxideToElementFactor(el)

                        sum += ppmd(analyte)[csi]*f

                    norm[csi] = (criteria['value']*1e4)/sum
                    crit_index[csi] = ci

            else:
                if not isvalue:
                    continue

                sum = np.zeros(len(si))
                for el in iselement.split(','):
                    if 'oxide' in isunit:
                        actual_el = data.timeSeries(el).property('Element')
                        try:
                            oxide_forms = sel.property('OxideForms')
                            matches = [f for f in oxide_forms.split(',') if actual_el in f]
                            f = data.oxideToElementFactor(matches[0])
                        except:
                            f = data.oxideToElementFactor(actual_el)
                    else:
                        f = 1

                    f *= 0.0001 if 'wtpc' in isunit else 1

                    if np.isnan(f):
                        print(f'Not including {el} in sum as it has a NaN f value.')
                        continue

                    try:
                        ppmCh = data.timeSeries(f'{el}_ppm')
                    except:
                        print(f'There was an issue using {el} ppm channel as an internal standard and it has been skipped')
                        continue

                    if np.nanmedian(data.timeSeries(f'{el}_ppm').data()[si]*f) < 0 :
                        print(f'\nNot including {el} in sum as it has a negative median concentration {np.nanmedian(data.timeSeries(f"{el}_ppm").data()[si])} over the course of selection {sel.name}.\n')
                        continue

                    try:
                        sum += data.timeSeries(f'{el}_ppm').data()[si]*f
                    except:
                        print(f'There was an issue using {el} ppm channel as an internal standard and it has been skipped')

                norm[si] = float(isvalue)/(sum)
                if np.sum(sum) < 0:
                    print(f'Negative sum for {sel.name}: {np.sum(sum)}')

        # Replace parts that end up inf with 1
        norm[np.abs(norm) == np.inf] = 1

        data.createTimeSeries('CriteriaIndex', data.Intermediate, indexChannel.time(), crit_index, commonProps)

        for c in data.timeSeriesList(data.Output):
            d = c.data()*norm
            c.setData(d)

        # Not going to use the built in drs yield calculator because it doesn't handle
        # multi element the best and also doesn't handle criteria
        data.createTimeSeries('RelativeYield', data.Intermediate, indexChannel.time(), 1/norm, commonProps)


    data.updateResults()

    # This bit of code uses the residual of its nearest RM to apply an additional correction
    # Todo: make it configurable
    use_fg = drs.setting('UseFG')
    aff_on = float(drs.setting('AffinityCorrection%'))/100.0

    if drs.setting('AffinityCorrection'):
        drs.message.emit('Doing affinity correction')
        drs.progress.emit(92)
        extFactors = {}
        for ext in externalsInUse:
            extFactors[ext] = {}
            for ppmc in data.timeSeriesList(data.Output, {'Units': 'µg.g-1'}):
                try:
                    rm = data.referenceMaterialData(ext)[ppmc.property('Element')].valueInUnits('fg') if use_fg else data.referenceMaterialData(ext)[ppmc.property('Element')].valueInPPM()
                    meas = data.groupResult(data.selectionGroup(ext), ppmc).value()
                    extFactors[ext][ppmc.name] = meas/rm
                    if abs(extFactors[ext][ppmc.name]-1) > aff_on:
                        print(f'Not going to correct {ppmc.name} for {ext} due to' \
                        ' its large relative difference from the accepted value: ' \
                        f'{abs(extFactors[ext][ppmc.name]-1):.2f}')
                        extFactors[ext][ppmc.name] = 1.
                        continue
                    else:
                        print(f'Will apply an affinity correction ' \
                              f'of {extFactors[ext][ppmc.name] * 100.:.2f}% for {ppmc.name} on groups related to {ext}')
                except Exception as e:
                    print(e)

        print(extFactors)

        for sel in sels:
            if sel.group().name in externalsInUse:
                continue
            for ppmc in data.timeSeriesList(data.Output, {'Units': 'µg.g-1'}):
                d = ppmc.data()
                try:
                    f = extFactors[sel.property('External affinity')][ppmc.name]
                    ind = ppmc.selectionIndices(sel)
                    d[ind] =  d[ind]/f
                    ppmc.setData(d)
                except Exception as e:
                    pass
                    #print(e)

    data.updateResults()

    # Store sensitivities so LODs can be determined by results manager
    drs.message.emit('Storing sensitivities')
    drs.progress.emit(95)
    drs.storeExternalSensitivities()
    # badSels = []
    # for sel in sels:
    #     badChannels = []
    #     for cps in data.timeSeriesList(data.Intermediate):
    #         if 'CPS' not in cps.name or 'TotalBeam' in cps.name:
    #             continue
    #         try:
    #             ppm = data.timeSeries(cps.name.replace('CPS', 'ppm'))
    #             s = data.result(sel, cps).value()/data.result(sel, ppm).value()
    #             sel.setProperty('Sensitivity %s'%(cps.name.replace('_CPS', '')), s)
    #         except:
    #             badChannels.append(cps.name.replace('CPS', 'ppm'))
    #             continue
    #
    #     badSels.append(sel)

    #badChannelsString = ', '.join(badChannels)
    #badChannelsString = badChannelsString if len(badChannelsString) < 25 else badChannelsString[0:23]+'...'
    #badSelsString = ', '.join([s.name for s in badSels])
    #badSelsString = badSelsString if len(badSelsString) < 25 else badSelsString[0:23]+'...'
    #IoLog.warning(f'There was a problem calculating the sensitivity of {badChannelsString} for selection(s) {badSelsString}')

    # Convert ppm output channel to wtpc if user has set this setting
    if QSettings().value('ConvertPPMtoWtPCOutputChannels', False):
        # We want to convert to wtpc if:
        # Using a master external + the maser is in wtpc, or
        # Not using a master extrnal but all of the RMs in use are in wtpc.

        def allInWtpc(channel):
            channelProperties = channel.properties()
            if not 'Reference Material' in channelProperties or not 'Element' in channelProperties:
                return False

            element = channelProperties['Element']
            rms = channelProperties['Reference Material'].split(',')
            return all([element in data.referenceMaterialData(rm) and data.referenceMaterialData(rm)[element].units() == 'wtpc' for rm in rms])

        def convertChannelToWtpc(channel):
            print('Converting ' + ch.name + ' from ppm to wtpc...')
            ch.setData(ch.data() / 10000.0)
            ch.name = ch.name.replace('ppm', 'wtpc')
            ch.setProperty('Units', 'wtpc')

        masterGroupName = drs.setting('MasterExternal')
        normalizeExternals = drs.setting('NormalizeExternals')

        for ch in data.timeSeriesList(data.Output):
            if allInWtpc(ch):
                convertChannelToWtpc(ch)
                continue

            try:
                element = ch.property('Element')
                res = data.referenceMaterialData(masterGroupName)[element]
                if res.units() == 'wtpc':
                    convertChannelToWtpc(ch)
            except:
                continue

    for ts in data.timeSeriesList(data.Intermediate):
        if ts.name.endswith('_CPS'):
            print(f"Removing {ts.name} from Intermediate")
            data.removeTimeSeries(ts.name)

    for ts in data.timeSeriesList(data.Intermediate):
        if ts.name.endswith('_slope'):
            print(f"Removing {ts.name} from Intermediate")
            data.removeTimeSeries(ts.name)

    for ts in data.timeSeriesList(data.Intermediate):
        if ts.name.endswith('_intercept'):
            print(f"Removing {ts.name} from Intermediate")
            data.removeTimeSeries(ts.name)

    # Need to update results again to get LODs calculated
    data.updateResults()
    data.addCitationToSession('ThreeDTE')
    drs.message.emit("Finished!")
    drs.progress.emit(100)
    drs.finished.emit()


'''
GUI-related classes
'''
class ExternalsModel(QAbstractTableModel):

    throughZeroChanged = Signal()
    fractionationChanged = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.refreshChannels()

    def refreshChannels(self):
        self.beginResetModel()
        self.channels = [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name and 'AllLight' not in c.name]

        try:
            drs.baselineSubtract(None, data.timeSeriesList(data.Input), None, 0, 0)
            for c in data.timeSeriesList(data.Input):
                if not c.property('Model'):
                    c.setProperty('Model', 'ODR')
        except:
            print('You must import data and create selections before using the 3DTE DRS.')


        self.endResetModel()

    def updateData(self):
        self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, 1))

    def rowCount(self, index=QModelIndex()):
        return len(self.channels)

    def columnCount(self, index):
        return 5

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            if section == 0:
                return 'Channel'
            elif section == 1:
                return 'External standards'
            elif section == 2:
                return 'Model'
            elif section == 3:
                return 'Zero'
            elif section == 4:
                return 'Frac. correction'

        return None

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.UserRole:
            return self.channels[index.row()]

        if role == Qt.CheckStateRole and index.column() >= 3:
            if index.column() == 3 and self.channels[index.row()].property('FitThroughZero'):
                return Qt.Checked
            elif index.column() == 4 and self.channels[index.row()].property('FractionationCorrection'):
                return Qt.Checked
            return Qt.Unchecked

        if role != Qt.DisplayRole:
            return None

        if index.column() == 0:
            return self.channels[index.row()].name
        elif index.column() == 1:
            return self.channels[index.row()].property('External standard')
        elif index.column() == 2:
            return self.channels[index.row()].property('Model')
        elif index.column() == 4:
            ft = self.channels[index.row()].property('FractionationFitType')
            if not ft:
                ft = 'None'
            return ft

        return None

    def setData(self, index, value, role = Qt.EditRole):
        if role == Qt.CheckStateRole and index.column() == 3:
            self.channels[index.row()].setProperty('FitThroughZero', value == Qt.Checked)
            self.throughZeroChanged.emit()
            self.dataChanged.emit(index, index)
        if role == Qt.CheckStateRole and index.column() == 4:
            self.channels[index.row()].setProperty('FractionationCorrection', value == Qt.Checked)
            ft = self.channels[index.row()].property('FractionationFitType')
            if value == Qt.Checked and (not ft or ft == 'None'):
                self.channels[index.row()].setProperty('FractionationFitType', 'Linear')
            elif value == Qt.Unchecked:
                self.channels[index.row()].setProperty('FractionationFitType', 'None')

            self.dataChanged.emit(index, index)
            self.fractionationChanged.emit()

    def flags(self, index):
        if index.column() in [1, 2, 3, 4]:
            return QAbstractTableModel.flags(self, index) | Qt.ItemIsEditable | Qt.ItemIsUserCheckable
        return QAbstractTableModel.flags(self, index)

class ExternalsDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        QStyledItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        if index.column() == 1:
            return ReferenceMaterialComboBox(parent, index.data(Qt.UserRole))
        elif index.column() == 2:
            cb = QComboBox(parent)
            cb.addItems(['ODR', 'OLS', 'WLS', 'RLM', 'York'])
            return cb
        elif index.column() == 4:
            cb = QComboBox(parent)
            cb.addItems(['None', 'Linear', 'Spline'])
            return cb

        return QStyledItemDelegate.createEditor(self, parent, option, index)

    def setEditorData(self, editor, index):
        if index.column() == 1:
            editor.clear()
            editor.addItem(index.data(Qt.UserRole).property('External standard'))
            editor.currentText = index.data(Qt.UserRole).property('External standard')
        elif index.column() == 2:
            editor.currentText = index.data(Qt.UserRole).property('Model')
        elif index.column() == 4:
            editor.currentText = index.data(Qt.UserRole).property('FractionationFitType')

    def setModelData(self, editor, model, index):
        if index.column() == 2:
            index.data(Qt.UserRole).setProperty('Model', editor.currentText)
        elif index.column() == 4:
            index.data(Qt.UserRole).setProperty('FractionationCorrection', 'None' != editor.currentText)
            index.data(Qt.UserRole).setProperty('FractionationFitType', editor.currentText)
        try:
            model.sourceModel.dataChanged.emit(index, index)
        except:
            model.dataChanged.emit(index, index)


class ReferenceMaterialComboBox(QComboBox):

    def __init__(self, parent, channel):
        QComboBox.__init__(self, parent)
        self.channel = channel

    def updateText(self):
        self.clear()
        self.addItem(self.channel.property('External standard'))
        self.currentText = self.channel.property('External standard')

    def showPopup(self):
        menu = ReferenceMaterialsMenu(self)
        menu.activeChannels = [self.channel.name]
        menu.rmsForActiveChannels = []
        if type(self.channel.property('External standard')) == str:
            menu.rmsForActiveChannels = self.channel.property('External standard').split(',')
        menu.rmsChanged.connect(self.updateText)
        menu.setFixedWidth(self.width)
        p = self.pos
        p += QPoint(0, self.height)
        menu.exec_(self.parent().mapToGlobal(p))

class ReferenceMaterialsMenu(QMenu):

    rmsChanged = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.activeChannels = []
        self.rmsForActiveChannels = []

        for rm in data.referenceMaterialNames():
            a = QWidgetAction(self)
            cb = QCheckBox(rm, self)
            cb.setStyleSheet('QCheckBox { padding-left: 5px; margin: 3px; }')
            a.setDefaultWidget(cb)
            self.addAction(a)
            cb.clicked.connect(partial(self.updateChannels, rm))

        self.addSeparator()
        modelAction = self.addAction('Model')
        modelAction.triggered.connect(self.setChannelsToModel)

        self.aboutToShow.connect(self.updateMenu)

    def updateMenu(self):
        for a in self.actions():
            try:
                cb = a.defaultWidget()
                cb.setChecked(False)
                if cb.text in list(itertools.chain.from_iterable([rms.split(',') for rms in self.rmsForActiveChannels])):
                    cb.setChecked(True)
            except Exception as e:
                print(e)

    def updateChannels(self, rmName, b):
        for c in [data.timeSeries(cn) for cn in self.activeChannels]:
            try:
                rms_for_ch = c.property('External standard').split(',') if c.property('External standard') else []
            except:
                rms_for_ch = []

            if 'Model' in rms_for_ch:
                rms_for_ch = []

            if b:
                rms_for_ch = list(set(rms_for_ch + [rmName]))
            else:
                rms_for_ch = list(filter(lambda rm: rm != rmName, rms_for_ch))

            c.setProperty('External standard', ','.join(rms_for_ch))

        self.rmsChanged.emit()

    def setChannelsToModel(self):
        for c in [data.timeSeries(cn) for cn in self.activeChannels]:
            c.setProperty('External standard', 'Model')

        self.rmsChanged.emit()


class InternalsDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        QStyledItemDelegate.__init__(self, parent)

    def createEditor(self, parent, option, index):
        if index.column() == 2:
            return ChannelsComboBox(parent, index.data(Qt.UserRole))
        elif index.column() == 4:
            cb = QComboBox(parent)
            cb.addItems(['ppm', 'ppb', 'wtpc', 'wtpc_oxide'])
            return cb

        return QStyledItemDelegate.createEditor(self, parent, option, index)

    def setEditorData(self, editor, index):
        if index.column() == 2:
            editor.clear()
            editor.addItem(index.data(Qt.UserRole).property('Internal element'))
            editor.currentText = index.data(Qt.UserRole).property('Internal element')
        elif index.column() == 3:
            editor.text = str(index.data(Qt.UserRole).property('Internal value'))
        elif index.column() == 4:
            editor.currentText = index.data(Qt.UserRole).property('Internal units')

    def setModelData(self, editor, model, index):
        if index.column() in [2,4]:
            index.model().setData(index, editor.currentText) # These are QComboBox
        elif index.column() in [3]:
            index.model().setData(index, float(editor.text)) # These are QLineEdit

class InternalsModel(QAbstractTableModel):

    def __init__(self, parent):
        super().__init__(parent)
        self.refreshSelections()

    def refreshSelections(self):
        self.beginResetModel()
        self.selections = [[s for s in sg.selections()] for sg in data.selectionGroupList(data.ReferenceMaterial | data.Sample)]
        self.selections = list(itertools.chain.from_iterable(self.selections))
        self.endResetModel()

    def updateData(self, selections=None):
        # This is triggered when the channels are changed
        if selections is None:
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount()-1, self.columnCount()-1))
            return

        for s in selections:
            try:
                is_names = s.property('Internal element').split(',')
            except:
                pass

            row = self.selections.index(s)

            if 'Criteria' in is_names:
                self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()-1))
                continue

            elements = [data.timeSeries(name).property('Element') for name in is_names if name]
            units = s.property('Internal units')

            if not units:
                units = 'ppm'
                s.setProperty('Internal units', units)

            try:
                if s.group().type == data.Sample:
                    sum = np.sum([Result(float(s.property(e)), 0, 'wtpc', e).valueInUnits(units) for e in elements])
                elif s.group().type == data.ReferenceMaterial:
                    sum = np.sum([data.referenceMaterialData(s.group().name)[e].valueInUnits(units) for e in elements if e in data.referenceMaterialData(s.group().name)])
            except:
                sum = float(s.property('Internal value'))

            s.setProperty('Internal value', sum)
            self.dataChanged.emit(self.index(row, 0), self.index(row, self.columnCount()-1))

    def rowCount(self, parent=QModelIndex()):
        return len(self.selections)

    def columnCount(self, parent=QModelIndex()):
        return 6

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            if section == 0:
                return 'Group'
            elif section == 1:
                return 'Selection'
            elif section == 2:
                return 'Element'
            elif section == 3:
                return 'Value'
            elif section == 4:
                return 'Units'
            elif section == 5:
                return 'Affinity'

        return None

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.UserRole:
            return self.selections[index.row()]

        if role == Qt.ToolTipRole:
            return self.data(index, Qt.DisplayRole)

        if role != Qt.DisplayRole:
            return None

        if index.column() == 0:
            return self.selections[index.row()].group().name
        elif index.column() == 1:
            return self.selections[index.row()].name
        elif index.column() == 2:
            return self.selections[index.row()].property('Internal element')
        elif index.column() == 3:
            if self.selections[index.row()].property('Internal element') == 'Criteria':
                return '-'
            return self.selections[index.row()].property('Internal value')
        elif index.column() == 4:
            if self.selections[index.row()].property('Internal element') == 'Criteria':
                return '-'
            return self.selections[index.row()].property('Internal units')
        elif index.column() == 5:
            s = self.selections[index.row()]
            afe = s.property('Affinity elements')
            ext = s.property('External affinity')
            if ext:
                return f'{ext} (from {afe})'

            return f'{afe}'

        return None

    def setData(self, index, value, role = Qt.EditRole):
        if role == Qt.EditRole and index.column() == 2:
            index.data(Qt.UserRole).setProperty('Internal element', value)
            self.dataChanged.emit(index, index)
        elif role == Qt.EditRole and index.column() == 3:
            index.data(Qt.UserRole).setProperty('Internal value', value)
            self.dataChanged.emit(index, index)
        elif role == Qt.EditRole and index.column() == 4:
            index.data(Qt.UserRole).setProperty('Internal units', value)
            self.dataChanged.emit(index, index)

    def flags(self, index):
        if index.column() > 1:
            return QAbstractTableModel.flags(self, index) | Qt.ItemIsEditable
        return QAbstractTableModel.flags(self, index)


class ChannelsComboBox(QComboBox):

    def __init__(self, parent, selection):
        QComboBox.__init__(self, parent)
        self.selection = selection

    def updateText(self):
        self.clear()
        self.addItem(self.selection.property('Internal element'))
        self.currentText = self.selection.property('Internal element')

    def showPopup(self):
        menu = ChannelsMenu(self)
        menu.setSelections([self.selection])
        menu.setFixedWidth(self.width)
        menu.channelsChanged.connect(self.updateText)
        p = self.pos
        p += QPoint(0, self.height)
        menu.exec_(self.parent().mapToGlobal(p))

class ChannelsMenu(QMenu):

    channelsChanged = Signal(list)

    def __init__(self, parent, propName='Internal element'):
        super().__init__(parent)
        self.propName = propName

        for channel in [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]:
            a = QWidgetAction(self)
            cb = QCheckBox(channel.name, self)
            cb.setStyleSheet('QCheckBox { padding-left: 5px; margin: 3px; }')
            a.setDefaultWidget(cb)
            self.addAction(a)
            cb.clicked.connect(partial(self.updateSelections, channel.name))

        self.addSeparator()
        selectAll = self.addAction('Select all')
        selectAll.triggered.connect(self.selectAll)
        selectNone = self.addAction('Select none')
        selectNone.triggered.connect(self.selectNone)
        self.addSeparator()
        crit = self.addAction('Use criteria')
        crit.triggered.connect(self.setSelectionsForCriteria)

        self.aboutToShow.connect(self.updateMenu)
        self.selections = []
        self.channels = []

    def setSelections(self, sels):
        self.selections = sels

    def updateMenu(self):
        try:
            s = self.selections[0]
            chs = s.property(self.propName).split(',')
            self.setChannels(chs)
        except Exception as e:
            print(e)

    def selectAll(self):
        self.channels = [c for c in data.timeSeriesNames(data.Input) if 'TotalBeam' not in c]
        for s in self.selections:
            s.setProperty(self.propName, ','.join(self.channels))
        self.channelsChanged.emit(self.selections)

    def selectNone(self):
        for s in self.selections:
            s.setProperty(self.propName, '')
        self.channels = []

        # If this is resetting the external affinities, also reset the Ext Affinity
        if self.propName == 'Affinity elements':
            for s in self.selections:
                s.setProperty('External affinity', '')

        self.channelsChanged.emit(self.selections)

    def setSelectionsForCriteria(self):
        for s in self.selections:
            s.setProperty(self.propName, 'Criteria')

        self.channelsChanged.emit(self.selections)

    def setChannels(self, channels):
        self.channels = channels

        for a in self.actions():
            try:
                a.defaultWidget().setChecked(a.defaultWidget().text in self.channels)
            except Exception as e:
                continue

    def updateSelections(self, channelName, b):

        self.channels = []
        for a in self.actions():
            try:
                if a.defaultWidget().isChecked():
                    self.channels.append(a.defaultWidget().text)
            except Exception as e:
                continue

        for s in self.selections:
            if s.property(self.propName):
                sie = s.property(self.propName).split(',')
                sie[:] = (v for v in sie if v != 'Criteria' and v not in data.referenceMaterialNames())
            else:
                sie = []
            if channelName not in sie and b: sie.append(channelName)
            if channelName in sie and not b: sie.remove(channelName)
            s.setProperty(self.propName, ','.join(sie))

        self.channelsChanged.emit(self.selections)


class CriteriaModel(QAbstractTableModel):

    def __init__(self, criteria, parent):
        QAbstractTableModel.__init__(self, parent)
        self.criteria = criteria

    def setCriteria(self, criteria, b=None): # note: the b is just to make connections from action.triggered happy
        if not criteria:
            criteria = []

        self.beginResetModel()
        self.criteria = list(criteria)
        self.endResetModel()

    def rowCount(self, index=QModelIndex()):
        return len(self.criteria)

    def addRow(self):
        self.beginInsertRows(QModelIndex(), self.rowCount(), self.rowCount())
        self.criteria.append({
            'name': 'Unnamed',
            'criteria': '',
            'analytes': '',
            'value': 100.,
            'oxides': True,
            'oxide_forms': []
        })
        self.endInsertRows()

    def removeRow(self, row):
        self.beginRemoveRows(QModelIndex(), row, row)
        del self.criteria[row]
        self.endRemoveRows()

    def moveRow(self, row, dir):
        if row == 0 and dir < 0:
            return
        elif row == self.rowCount() - 1 and dir > 0:
            return

        dirmod = 0 if dir < 0 else 1

        if self.beginMoveRows(QModelIndex(), row, row, QModelIndex(), row+dir+dirmod):
            self.criteria[row], self.criteria[row+dir] = self.criteria[row+dir], self.criteria[row]
            self.endMoveRows()

    def columnCount(self, index):
        return 6

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            if section == 0:
                return 'Name'
            elif section == 1:
                return 'Criteria'
            elif section == 2:
                return 'Analytes'
            elif section == 3:
                return 'Value'
            elif section == 4:
                return 'Oxides'
            elif section == 5:
                return 'Oxide forms'

        return None

    def data(self, index, role):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole or role == Qt.EditRole:
            if index.column() == 0:
                return self.criteria[index.row()]['name']
            elif index.column() == 1:
                return self.criteria[index.row()]['criteria']
            elif index.column() == 2:
                return self.criteria[index.row()]['analytes']
            elif index.column() == 3:
                return float(self.criteria[index.row()]['value'])
            elif index.column() == 4:
                return self.criteria[index.row()]['oxides']
            elif index.column() == 5:
                return ','.join(self.criteria[index.row()]['oxide_forms'])
        elif role == Qt.CheckStateRole and index.column() == 4:
            return Qt.Checked if self.criteria[index.row()]['oxides'] else Qt.Unchecked


        return None

    def setData(self, index, value, role = Qt.EditRole):
        if role == Qt.CheckStateRole and index.column() == 4:
            self.criteria[index.row()]['oxides'] = value == Qt.Checked

        if index.column() == 0:
            self.criteria[index.row()]['name'] = value
        elif index.column() == 1:
            self.criteria[index.row()]['criteria'] = value
        elif index.column() == 2:
            self.criteria[index.row()]['analytes'] = value
        elif index.column() == 3:
            self.criteria[index.row()]['value'] = float(value)
        elif index.column() == 5:
            self.criteria[index.row()]['oxide_forms'] = str(value).split(',')

        self.dataChanged.emit(index, index)

    def flags(self, index):
        if index.column() == 4:
            return QAbstractTableModel.flags(self, index) | Qt.ItemIsUserCheckable

        if index.column() != 4:
            return QAbstractTableModel.flags(self, index) | Qt.ItemIsEditable

        return QAbstractTableModel.flags(self, index)

class CriteriaDialog(QDialog):

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle('Internal standard criteria')
        self.table = QTableView(self)
        self.table.setModel(CriteriaModel([],self.table))
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.contextMenuPolicy = Qt.CustomContextMenu

        def makeButton(text, icon):
            b = QToolButton(self)
            b.text = text
            b.setIcon(CUI().icon(icon))
            b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            return b

        self.addButton = makeButton('Add', 'plus')
        self.removeButton = makeButton('Remove', 'minus')
        self.upButton = makeButton('Up', 'arrowup')
        self.downButton = makeButton('Down', 'arrowdown')
        self.presetsButton = makeButton('Presets', 'book')
        self.bb = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel, self)

        topLayout = QHBoxLayout()
        topLayout.addWidget(self.addButton)
        topLayout.addWidget(self.removeButton)
        topLayout.addWidget(self.upButton)
        topLayout.addWidget(self.downButton)
        topLayout.addStretch(1)
        topLayout.addWidget(self.presetsButton)

        self.setLayout(QVBoxLayout())
        self.layout().addLayout(topLayout)
        self.layout().addWidget(self.table)
        self.layout().addWidget(self.bb)

        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        self.addButton.clicked.connect(self.table.model().addRow)
        self.removeButton.clicked.connect(lambda: self.table.model().removeRow(self.table.selectionModel().selectedRows()[0].row()))
        self.upButton.clicked.connect(lambda: self.table.model().moveRow(self.table.selectionModel().selectedRows()[0].row(), -1))
        self.downButton.clicked.connect(lambda: self.table.model().moveRow(self.table.selectionModel().selectedRows()[0].row(), 1))
        self.table.customContextMenuRequested.connect(self.showAnalytesMenu)

        self.setupPresetsMenu()

        self.resize(700, 500)

    def setupPresetsMenu(self):
        self.presetsMenu = QMenu(self)
        self.presetsButton.setMenu(self.presetsMenu)
        self.presetsButton.setPopupMode(QToolButton.InstantPopup)

        settings = QSettings()
        presets = dict(settings.value('3DTE/CriteriaPresets', {}))

        for k in presets:
            a = self.presetsMenu.addAction(k)
            a.setProperty('criteria', presets[k])
            a.triggered.connect(partial(self.table.model().setCriteria, presets[k]))

        self.presetsMenu.addSeparator()
        saveAction = self.presetsMenu.addAction('Save')
        saveAction.triggered.connect(self.savePreset)
        removeAction = self.presetsMenu.addAction('Remove')
        removeAction.triggered.connect(self.removePreset)

    def savePreset(self):
        name = QInputDialog.getText(self, 'Save criteria preset', 'Preset name')

        if not name:
            return

        presets = QSettings().value('3DTE/CriteriaPresets', {})
        presets[name] = self.table.model().criteria
        QSettings().setValue('3DTE/CriteriaPresets', presets)
        self.setupPresetsMenu()

    def removePreset(self):
        names = list(QSettings().value('3DTE/CriteriaPresets', {}).keys())
        name = QInputDialog.getItem(self, 'Remove criteria preset', 'Preset name', names)

        if not name:
            return

        presets = QSettings().value('3DTE/CriteriaPresets', {})
        del presets[name]
        QSettings().setValue('3DTE/CriteriaPresets', presets)
        self.setupPresetsMenu()

    def showAnalytesMenu(self, point):
        index = self.table.indexAt(point)
        if index.column() != 2:
            return

        menu = ChannelsMenu(self)
        menu.removeAction(menu.actions()[-1])
        menu.setChannels(index.data().split(','))
        menu.channelsChanged.connect(lambda: index.model().setData(index, ','.join(menu.channels)))
        menu.exec_(self.mapToGlobal(point))

    def criteria(self):
        return self.table.model().criteria


class BlockAssignmentsModel(QAbstractTableModel):

    def __init__(self, selections, parent):
        super().__init__(parent)
        self.selections = selections
        self.blocks = []
        self.selToBlockMap = {}

    def setBlocks(self, blocks):
        self.beginResetModel()
        self.blocks = blocks
        self.selToBlockMap = {}

        for block in self.blocks:
            for sel in block.selections:
                self.selToBlockMap[sel.property('UUID')] = block.label

        self.endResetModel()

    def reset(self):
        self.beginResetModel()
        self.setBlocks(self.blocks)
        self.endResetModel()

    def rowCount(self, index=QModelIndex()):
        return len(self.selections)

    def columnCount(self, index=QModelIndex()):
        return 4

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None

        if orientation == Qt.Horizontal:
            if section == 0:
                return 'Group'
            elif section == 1:
                return 'Selection'
            elif section == 2:
                return 'Time'
            elif section == 3:
                return 'Block'

        return None

    def data(self, index, role):
        if not index.isValid():
            return None

        s = self.selections[index.row()]

        if role == Qt.UserRole:
            return s

        try:
            blockIndex = self.selToBlockMap[s.property('UUID')]
        except:
            blockIndex = None

        if role == Qt.BackgroundColorRole:
            if not blockIndex:
                return None

            return colors[blockIndex]

        if role == Qt.DecorationRole and index.column() == 3:
            if s.property('Block') is not None:
                return CUI().icon('bullseye')

        if role != Qt.DisplayRole:
            return

        if index.column() == 0:
            return s.group().name
        elif index.column() == 1:
            return s.name
        elif index.column() == 2:
            if s.hasLinks():
                return s.linkedStartTime().toString('yyyy-MM-dd hh:mm:ss.zzz')
            else:
                return s.startTime.toString('yyyy-MM-dd hh:mm:ss.zzz')
        elif index.column() == 3:
            return blockIndex

        return None

class BlockPlot(QWidget):

    def __init__(self, parent):
        if parent is None:
            raise Exception("The 3D Traces Block Plot was initialised without a parent")

        super().__init__(parent)
        self.settingsWidget = parent
        self.setLayout(QVBoxLayout())

        self.buttonLayout = QHBoxLayout()
        self.blockDownButton = QToolButton(self)
        self.blockDownButton.setIcon(CUI().icon('arrowleft'))
        self.blockLabel = QLabel("")
        self.blockUpButton = QToolButton(self)
        self.blockUpButton.setIcon(CUI().icon('arrowright'))
        self.configButton = QToolButton(self)
        self.configButton.setIcon(CUI().icon('wrench'))
        self.configButton.setFixedSize(25, 25)
        self.configButton.clicked.connect(self.configBlocks)

        self.buttonLayout.addWidget(self.blockDownButton)
        self.buttonLayout.addWidget(self.blockLabel)
        self.buttonLayout.addWidget(self.configButton)
        self.buttonLayout.addWidget(self.blockUpButton)
        self.buttonLayout.insertStretch(1)
        self.buttonLayout.insertStretch(4)
        self.layout().addLayout(self.buttonLayout)
        self.blockDownButton.clicked.connect(lambda: self.setBlock(self.bn - 1))
        self.blockUpButton.clicked.connect(lambda: self.setBlock(self.bn + 1))
        self.blockDownButton.setDisabled(True)
        #self.setAutoFillBackground(True)
        self.plot = Plot(self)
        self.plot.setBackground(CUI().tabBackgroundColor())
        self.plot.setToolsVisible(False)

        self.plot.left().label = f'Intensity ({drs.setting("StatName")})'
        if drs.setting('UseFG'):
            self.plot.bottom().label = 'Mass (fg)'
        else:
            self.plot.bottom().label = 'Concentration'
        self.layout().addWidget(self.plot)

        self.logButton = OverlayButton(self.plot, 'BottomLeft', 0, 0)
        self.logButton.setCheckable(True)
        self.logButton.setText('10ⁿ')
        self.logButton.setFixedSize(25, 25)
        self.logButton.clicked.connect(self.toggleLog)
        self.plot.installEventFilter(self.logButton)
        self.isLog = False

        self.showLegend = False
        menu = self.plot.contextMenu()
        menu.addSeparator()
        la = menu.addAction('Legend')
        la.setCheckable(True)
        la.setChecked(self.showLegend)
        la.triggered.connect(self.toggleLegend)

        self.bn = 0

    def toggleLegend(self, b):
        self.showLegend = b
        self.updatePlot()

    def configBlocks(self):
        d = QDialog(self)
        d.setWindowTitle('Block assignments')
        d.setLayout(QVBoxLayout())
        d.resize(600, 600)

        topLayout = QHBoxLayout()
        setButton = QPushButton(d)
        setButton.setText('Set selected')
        topLayout.addWidget(setButton)
        clearButton = QPushButton(d)
        clearButton.setText('Clear selected')
        topLayout.addWidget(clearButton)
        saveAssignmentsButton = QPushButton(d)
        saveAssignmentsButton.setText('Save as assignments')
        topLayout.addWidget(saveAssignmentsButton)
        topLayout.addStretch()
        topLayout.addWidget(QLabel('Method', d))
        methodComboBox = QComboBox(d)
        methodComboBox.addItems(['Assigned', 'Simple', 'Clustering', 'Auto Clustering'])
        if drs.setting('BlockFindingMethod'):
            methodComboBox.setCurrentText(drs.setting('BlockFindingMethod'))
        else:
            methodComboBox.setCurrentText('Simple')
        nClustersSpinBox = QSpinBox(d)
        nClustersSpinBox.setMinimum(1)
        nClustersSpinBox.setMaximum(1000)
        try:
            nClusters = int(drs.setting('NClusters')) if int(drs.setting('NClusters')) > 0 else 5
            nClustersSpinBox.setValue(nClusters)
        except Exception as e:
            print(e)

        topLayout.addWidget(methodComboBox)
        topLayout.addWidget(nClustersSpinBox)
        nClustersSpinBox.setVisible(drs.setting('BlockFindingMethod') and drs.setting('BlockFindingMethod') == 'Clustering')
        methodComboBox.activated.connect(lambda: nClustersSpinBox.setVisible(methodComboBox.currentText == 'Clustering'))
        d.layout().addLayout(topLayout)

        table = QTableView(d)
        d.layout().addWidget(table)

        plot = Plot(d)
        g = plot.addGraph()
        tb = data.timeSeries('TotalBeam')
        g.setData(tb.time(), tb.data())
        plot.bottom().label = 'Time (s)'
        plot.left().label = 'TotalBeam'
        plot.setFixedHeight(200)
        plot.setToolsVisible(False)
        plot.bottom().setDateTime(True)
        plot.left().setLogarithmic(True)
        d.layout().addWidget(plot)

        bb = QDialogButtonBox(QDialogButtonBox.Close, d)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        d.layout().addWidget(bb)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        table.setSortingEnabled(True)
        table.setSelectionMode(table.ExtendedSelection)
        table.setSelectionBehavior(table.SelectRows)

        # Changing line below to loop because below will fail if not Ext Stds set for all channels
        #externalsInUse = set(list(itertools.chain(*[c.property('External standard').split(',') for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name])))
        externalsInUse = []
        for c in data.timeSeriesList(data.Input):
            if 'TotalBeam' in c.name:
                continue
            if type(c.property('External standard')) != str:
                print(c.name + " had no Ext Std set and so it's been skipped in creating list of Ext Stds")
                continue

            externalsInUse.append(c.property('External standard').split(','))

        externalsInUse = set(list(itertools.chain(*externalsInUse)))

        try:
            externalsInUse.remove('Model')
        except KeyError:
            pass

        try:
            groups = [data.selectionGroup(ext) for ext in externalsInUse if ext]
        except RuntimeError as e:
            regex = r"requested group (.*) doesn't exist"
            m = re.search(regex, str(e))
            if m.group(1):
                QMessageBox.information(self, 'Error getting Ref Mats', f'Please make sure that you have created a group for {m.group(1)} and that there are no spelling mistakes')
                return
            else:
                QMessageBox.information(self, 'Error getting Ref Mats', 'Please make sure all reference material groups exist. Here is the error message: ' + str(e))
                return

        selections = list(itertools.chain(*[sg.selections() for sg in groups]))

        # Create a list of component selections
        component_sels = list(itertools.chain(*[s.linkedSelections() for s in selections if s.hasLinks()]))

        # Now kick them out
        selections = list(filter(lambda s: s.property('UUID') not in component_sels, selections))

        selections.sort(key=lambda s: s.midTimestamp)
        selMidTimes = [s.midTimestamp if not s.isLinked() else s.linkedMidTimestamp() for s in selections]
        selMidTimes.sort()

        blockModel = BlockAssignmentsModel(selections, d)
        sortModel = QSortFilterProxyModel(d)
        sortModel.setSourceModel(blockModel)
        table.setModel(sortModel)

        def update():
            print('update...')
            blocks = findBlocks(drs.setting('BlockFindingMethod'))
            blockModel.setBlocks(blocks)

            # Update plot
            plot.clearItems()
            blockSelections = []

            for block in blocks:
                minTime = np.min([s.startTime.toMSecsSinceEpoch()/1000.0 for s in block.selections])
                maxTime = np.max([s.endTime.toMSecsSinceEpoch()/1000.0 for s in block.selections])
                rect = plot.addRect( (minTime, 1), (maxTime, 0), 'ptPlotCoords', 'ptPlotCoords', plot.bottom(), plot.right())
                c = colors[block.label]
                c.setAlpha(90)
                rect.brush = QBrush(c)

            plot.right().range = QCPRange(0, 1)
            plot.rescaleAxes()
            if drs.setting('UseFG'):
                self.plot.bottom().label = 'Weight (fg)'
            else:
                self.plot.bottom().label = 'Concentration'

            plot.replot()

        update()

        def setBlocks():
            block = QInputDialog.getInt(d, 'Block number', 'Block number:', 1, 0, 100, 1)

            if block is None:
                return

            sels = [index.data(Qt.UserRole) for index in table.selectionModel().selectedRows()]
            for sel in sels:
                print(f'Setting block number for {sel.name} to {block}')
                sel.setProperty('Block', block)

            update()

        setButton.clicked.connect(setBlocks)

        def clearAssignments():
            sels = [index.data(Qt.UserRole) for index in table.selectionModel().selectedRows()]
            for sel in sels:
                sel.setProperty('Block', None)

            update()

        clearButton.clicked.connect(clearAssignments)

        def saveAssignments():
            for sel in blockModel.selections:
                sel.setProperty('Block', blockModel.selToBlockMap[sel.property('UUID')])

        saveAssignmentsButton.clicked.connect(saveAssignments)

        nClustersSpinBox.valueChanged.connect(lambda v: drs.setSetting('NClusters', v))
        nClustersSpinBox.valueChanged.connect(update)
        methodComboBox.activated.connect(lambda: drs.setSetting('BlockFindingMethod', methodComboBox.currentText))
        methodComboBox.activated.connect(update)

        d.exec_()
        self.settingsWidget.calibration.updateBlocks()
        self.settingsWidget.processExtSelection()

    def setBlock(self, bn):
        if bn < 0 or bn >= len(self.settingsWidget.calibration.blocks):
            print(f'Tried to set block to an invalid number... {bn}')
            return

        self.blockDownButton.setEnabled(bn > 0 and len(self.settingsWidget.calibration.blocks) > 0)
        self.blockUpButton.setEnabled(bn < (len(self.settingsWidget.calibration.blocks) - 1) and len(self.settingsWidget.calibration.blocks) > 0)

        self.bn = bn
        self.updatePlot()

    def toggleLog(self, b):
        self.plot.left().setLogarithmic(b)
        self.plot.bottom().setLogarithmic(b)
        self.isLog = b
        if self.isLog:
            self.plot.bottom().range = QCPRange(0.001, self.x_max*5)
            self.plot.left().range = QCPRange(1, self.y_max*5)
        else:
            self.plot.bottom().range = QCPRange(0, self.x_max*1.1)
            self.plot.left().range = QCPRange(0, self.y_max*1.1)
        self.plot.replot()

    def updatePlot(self):
        self.plot.clearGraphs()
        self.plot.clearItems()

        self.plot.left().label = f'Intensity ({drs.setting("StatName")})'
        if drs.setting('UseFG'):
            self.plot.bottom().label = 'Mass (fg)'
        else:
            self.plot.bottom().label = 'Concentration'

        channel = self.settingsWidget.selectedChannelNames[0]
        if len(self.settingsWidget.calibration.blocks) < 1:
            print('No blocks to update BlockPlot with...')
            return

        extStd = data.timeSeries(channel).property('External standard')
        if type(extStd) != str or (extStd.split(',')[0] not in data.selectionGroupNames() and extStd != 'Model'):
            print('Invalid external standard so not going to update plot.')
            self.plot.rescaleAxes()
            self.plot.replot()
            return

        # First, make sure we have RM values for this channel:
        if f'{channel}_RMppm' not in self.settingsWidget.calibration.blocks[0].dataFrame().columns:
            print(f'No valid values for reference materials for {channel}')
            self.plot.rescaleAxes()
            self.plot.replot()
            return

        this_df = self.settingsWidget.calibration.blocks[0].dataFrame()[f'{channel}_RMppm']
        if not this_df.notna().values.any() and extStd != 'Model':
            print(f'No valid values for reference materials for {channel}')
            self.plot.rescaleAxes()
            self.plot.replot()
            return

        # Make sure that the axes have the same extents for all blocks
        # Makes it easier to see differences
        # First though, make sure we have some values for each block
        # otherwise we'll get an error when trying to find the max/min
        # This might be caused by masking....
        for b in self.settingsWidget.calibration.blocks:
            if np.all(np.isnan(b.dataFrame()[f'{channel}'])):
                QMessageBox.information(self, 'No valid values', f'No valid values for {channel} in block {b.label}. Please check the data for this block.')
                print(f'No valid values for {channel} in block {b.label}')
                self.plot.rescaleAxes()
                self.plot.replot()
                return

        self.y_max = np.nanmax([np.nanmax(b.dataFrame()[f'{channel}']) for b in self.settingsWidget.calibration.blocks])
        self.y_min = np.nanmin([np.nanmin(b.dataFrame()[f'{channel}']) for b in self.settingsWidget.calibration.blocks])

        slopes = [b.slope(channel) for b in self.settingsWidget.calibration.blocks]
        for i, s in enumerate(slopes):
            if s is None:
                slopes[i] = np.nan

        if np.all(np.isnan(slopes)):
            print(f'No valid values for reference materials for {channel}')
            self.plot.rescaleAxes()
            self.plot.replot()
            return

        self.x_max = 0
        if extStd == 'Model':
            self.x_max = np.nanmax([self.y_max/b.slope(channel) for b in self.settingsWidget.calibration.blocks])
        else:
            for b in self.settingsWidget.calibration.blocks:
                maxRMvals = []
                df = b.dataFrame()
                if f'{channel}_RMppm' not in df.columns:
                    maxRMvals.append(np.nan)
                else:
                    try:
                        maxRMvals.append(np.nanmax(df[f'{channel}_RMppm']))
                    except:
                        print(df.to_string())

                self.x_max = max(self.x_max, np.nanmax(maxRMvals))

            #self.x_max = np.nanmax([np.nanmax(b.dataFrame()[f'{channel}_RMppm']) for b in self.settingsWidget.calibration.blocks])

        if np.isnan(self.x_max):
            print(f'No valid values for reference materials for {channel}')
            self.plot.rescaleAxes()
            self.plot.replot()
            return

        block = self.settingsWidget.calibration.blocks[self.bn]
        df = block.dataFrameForChannel(channel)
        # Users may have created their own RM files, and put 0 in for missing values.
        # This will mess with calculations below, so remove those here

        try:
            df = df[df[f'{channel}_RMppm'] != 0]

            slope = block.slope(channel)
            intercept = block.intercept(channel)

            if np.isnan(slope) or np.isnan(intercept):
                raise Exception('Slope or intercept is nan')

            ss_res = np.sum( (df[channel] - (slope*df[f'{channel}_RMppm'] + intercept))**2 )
            ss_tot = np.sum( (df[channel] - np.mean(df[channel]))**2 )

            #Note, for modeled fits, the r_sq value will always be 1. This is a little misleading,
            # so will manually change this below, when the annotation is added to the plot
            r_sq = 1 - (ss_res / ss_tot) if len(df) > 1 else 1
        except Exception as ex:
            self.blockLabel.setText(f"Block {self.bn + 1}/{len(self.settingsWidget.calibration.blocks)}")
            self.plot.annotate(f'<p style="color:black;">{channel} not calibrated in block {self.bn+1}</p>', 0.01, 0.01, 'ptAxisRectRatio', Qt.AlignLeft | Qt.AlignTop)
            self.plot.replot()
            return

        try:
            x_vals = np.logspace(-3, ceil(log10(self.x_max)), 200)
        except:
            x_vals = np.logspace(-3, 3, 200)
            self.x_max = 1000.

        y_vals = slope * x_vals + intercept

        #Add error envelope
#        res = block.fit(channel)['sm_res']
#        try:
#            if data.timeSeries(channel).property('FitThroughZero'):
#                pred = res.get_prediction(x_vals)
#            else:
#                pred = res.get_prediction(sm.add_constant(x_vals))
#            frame = pred.summary_frame(alpha=0.05)

#            print(frame.obs_ci_upper)
#            pg = self.plot.addGraph()
#            pg.setData(x_vals, frame.obs_ci_upper)

#            mg = self.plot.addGraph()
#            mg.setData(x_vals, frame.obs_ci_lower)
#        except Exception as e:
#            print(f'Could not use wls_prediction_std... {e}')

        # Use actual graph with lots of data points so it looks right in log scale, straight line DOES NOT
        g_cal = self.plot.addGraph()
        g_cal.setData(x_vals, y_vals)
        g_cal.setColor(Qt.red)
        g_cal.removeFromLegend()

        grad = QCPColorGradient('gpJet')
        symbols = [
            'ssCross', 'ssPlus', 'ssCircle', 'ssSquare', 'ssDiamond', 'ssStar', 'ssTriangle',
            'ssTriangleInverted', 'ssCrossSquare', 'ssPlusSquare', 'ssCrossCircle', 'ssPlusCircle'
            ]

        for gi, groupName in enumerate(df['group'].unique()):
            gdf = df[df['group'] == groupName]
            g_data = self.plot.addGraph()
            g_data.setData(gdf[f'{channel}_RMppm'], gdf[channel])
            color = grad.color(gi, 0, len(df['group'].unique()))
            g_data.setScatterStyle(symbols[gi%len(symbols)], 8, color, color)
            g_data.setLineStyle('lsNone')
            g_data.setColor(color)
            g_data.setName(groupName)

            eby = QCPErrorBars(self.plot.bottom(), self.plot.left())
            eby.setData(gdf[f'{channel}_Uncert'])
            eby.setDataPlottable(g_data)
            eby.errorType = QCPErrorBars.etValueError
            eby.removeFromLegend()

            ebx = QCPErrorBars(self.plot.bottom(), self.plot.left())
            ebx.setData(gdf[f'{channel}_RMppm_Uncert'])
            ebx.setDataPlottable(g_data)
            ebx.errorType = QCPErrorBars.etKeyError
            ebx.removeFromLegend()

        # Add slope and intercept annotation here:
        ann = self.plot.annotate('', 0.01, 0.01, 'ptAxisRectRatio', Qt.AlignLeft | Qt.AlignTop)
        if extStd == 'Model':
            ann.text = '''
                    <p style="color:black;">
                    <b>Slope</b>: %s<br>
                    <b>Intercept</b>: %s<br>
                    <b><i>R²</i></b> : N/A</p>'''%(formatResult(slope, block.fit(channel)['slope_uncert'])[0], formatResult(intercept, block.fit(channel)['intercept_uncert'])[0])

            self.plot.annotate('*Modeled', 0.9, 0.9, 'ptAxisRectRatio', Qt.AlignCenter, Qt.AlignCenter, False)

        else:
            ann.text = '''
                    <p style="color:black;">
                    <b>Slope</b>: %s<br>
                    <b>Intercept</b>: %s<br>
                    <b><i>R²</i></b> : %.3f</p>'''%(formatResult(slope, block.fit(channel)['slope_uncert'])[0], formatResult(intercept, block.fit(channel)['intercept_uncert'])[0], r_sq)

        if self.isLog:
            self.plot.bottom().range = QCPRange(0.001, self.x_max*5)
            self.plot.left().range = QCPRange(1, self.y_max*5)
        else:
            self.plot.bottom().range = QCPRange(0, self.x_max*1.1)
            self.plot.left().range = QCPRange(0, self.y_max * 1.1)

        self.plot.setLegendVisible(self.showLegend)
        self.plot.setLegendAlignment(Qt.AlignBottom | Qt.AlignRight)
        self.plot.replot()

        self.blockLabel.setText(f"Block {self.bn + 1}/{len(self.settingsWidget.calibration.blocks)}")


class FitsPlot(Plot):

    def __init__(self, parent):
        super().__init__(parent)
        self.settingsWidget = parent
        self.setToolsVisible(False)
        self.setBackground(CUI().tabBackgroundColor())

        self.left().label = f'Intensity ({drs.setting("StatName")})'
        if drs.setting('UseFG'):
            self.bottom().label = 'Mass (fg)'
        else:
            self.bottom().label = 'Concentration'
        self.grad = QCPColorGradient('gpViridis')

        self.setLegendAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.showLegend = False
        self.showColorScale = False
        self.blockCount = 0

        menu = self.contextMenu()
        menu.addSeparator()
        la = menu.addAction('Legend')
        la.setCheckable(True)
        la.setChecked(self.showLegend)
        la.triggered.connect(self.toggleLegend)

        ca = menu.addAction('Color scale')
        ca.setCheckable(True)
        ca.setChecked(self.showColorScale)
        ca.triggered.connect(self.toggleColorScale)

    def toggleLegend(self, b):
        self.showLegend = b
        self.setLegendVisible(b)
        self.replot()

    def toggleColorScale(self, b):
        self.showColorScale = b
        if not b:
            self.removeColorScale()
        else:
            self.addColorScale('Block number', self.grad, 1, self.blockCount)

        self.replot()


    def updatePlot(self):
        self.clearGraphs()
        self.clearItems()

        self.left().label = f'Intensity ({drs.setting("StatName")})'
        if drs.setting('UseFG'):
            self.bottom().label = 'Mass (fg)'
        else:
            self.bottom().label = 'Concentration'

        block_times = [block.midTime() for block in self.settingsWidget.calibration.blocks]
        channel = self.settingsWidget.selectedChannelNames[0]
        self.blockCount = len(self.settingsWidget.calibration.blocks)
        extStd = data.timeSeries(channel).property('External standard')
        if type(extStd) != str or (extStd.split(',')[0] not in data.selectionGroupNames() and extStd != 'Model'):
            self.replot()
            return

        for i, block in enumerate(self.settingsWidget.calibration.blocks):
            df = self.settingsWidget.calibration.blocks[i].dataFrameForChannel(channel)

            if extStd == 'Model':
                x_max = 100.
                self.annotate('*Modeled', 0.9, 0.9, 'ptAxisRectRatio', Qt.AlignCenter, Qt.AlignCenter, False)

            else:
                if f'{channel}_RMppm' not in df.columns:
                    print(f'No valid values for reference materials for {channel}')
                    #self.replot()
                    continue

                if not df[f'{channel}_RMppm'].notna().values.any():
                    #self.replot()
                    continue

                x_max = df[f'{channel}_RMppm'].dropna().max()
                x_max += x_max * 0.1

            if np.isnan(block.slope(channel)):
                print(f'Skipping block {i} because slope is nan!')
                continue

            slope = block.slope(channel)
            intercept = block.intercept(channel)

            x_vals = np.linspace(0, x_max)
            y_vals = slope * x_vals + intercept
            g = self.addGraph()
            g.setName(f'Block {block.label}')
            g.setColor(self.grad.color(i+1, 1, self.blockCount))
            g.setData(x_vals, y_vals)

        self.rescaleAxes()
        self.replot()


class FitParamsPlot(Plot):

    def __init__(self, parent):
        super().__init__(parent)
        self.settingsWidget = parent
        self.setToolsVisible(False)
        self.setBackground(CUI().tabBackgroundColor())
        self.left().label = 'Slope'
        self.right().label = 'Intercept'
        self.bottom().label = 'Time'
        self.bottom().setDateTime(True)
        self.top().label = 'Block'
        self.right().visible = True
        self.top().visible = True

        self.right().labelColor = QColor(Qt.red)
        self.left().labelColor = QColor(Qt.blue)

    def updatePlot(self):
        self.clearGraphs()
        self.clearItems()

        channel = self.settingsWidget.selectedChannelNames[0]
        extStd = data.timeSeries(channel).property('External standard')
        if type(extStd) != str or (extStd.split(',')[0] not in data.selectionGroupNames() and extStd != 'Model'):
            self.replot()
            return

        block_times = [block.midTime() for block in self.settingsWidget.calibration.blocks]

        if f'{channel}_RMppm' not in self.settingsWidget.calibration.blocks[0].dataFrame().columns:
            print(f'No valid values for reference materials for {channel}')
            self.replot()
            return

        if not self.settingsWidget.calibration.blocks[0].dataFrame()[f'{channel}_RMppm'].notna().values.any() and extStd != 'Model':
            self.replot()
            return

        slopes = [block.slope(channel) for block in self.settingsWidget.calibration.blocks]
        slopes_err = [block.slopeUncert(channel) for block in self.settingsWidget.calibration.blocks]
        inters = [block.intercept(channel) for block in self.settingsWidget.calibration.blocks]
        inters_err = [block.interceptUncert(channel) for block in self.settingsWidget.calibration.blocks]

        slopes_graph = self.addGraph(self.bottom(), self.left())
        slopes_eb = QCPErrorBars(self.bottom(), self.left())
        slopes_eb.setDataPlottable(slopes_graph)
        slopes_graph.setScatterStyle('ssCircle')

        inters_graph = self.addGraph(self.bottom(), self.right())
        inters_graph.setScatterStyle('ssDiamond', 6, Qt.red, Qt.red)
        inters_graph.setColor(Qt.red)
        inters_eb = QCPErrorBars(self.bottom(), self.right())
        inters_eb.setDataPlottable(inters_graph)

        slopes_graph.setData(block_times, slopes)
        slopes_eb.setData(slopes_err)

        inters_graph.setData(block_times, inters)
        inters_eb.setData(inters_err)

        self.rescaleAxes()
        self.top().range = QCPRange(1, len(block_times))
        self.bottom().scaleRange(1.1)
        self.left().scaleRange(1.1)
        self.top().scaleRange(1.1)

        if extStd == 'Model':
            self.left().label = 'Slope (Modeled)'
            self.right().label = 'Intercept (Modeled)'
        else:
            self.left().label = 'Slope'
            self.right().label = 'Intercept'

        self.replot()


class FractionationPlot(Plot):

    def __init__(self, parent):
        super().__init__(parent)
        self.settingsWidget = parent
        self.setToolsVisible(False)
        self.setBackground(CUI().tabBackgroundColor())
        self.bottom().label = 'Beam seconds'
        self.setLegendVisible(True)
        self.setLegendAlignment(Qt.AlignBottom | Qt.AlignRight)

        self.msg = OverlayMessage(
                    self, 'Warning',
                    'No internal standards set.',
                    0, 0.8, 40, 10
                    )
        self.msg.setCloseButtonVisible(False)
        self.msg.hide()

    def updatePlot(self):
        self.clearGraphs()
        self.clearItems()
        try:
            channel = self.settingsWidget.selectedChannelNames[0]
        except:
            return

        extStd = data.timeSeries(channel).property('External standard')

        if extStd == 'Model':
            text  = self.annotate('Fractionation correction not available\nfor channels with modelled sensitivity',
                          0.5, 0.5, 'ptAxisRectRatio', Qt.AlignCenter, Qt.AlignCenter, False)
            text.padding = QMargins(10,10,10,10)
            self.replot()
            return

        if type(extStd) != str or extStd.split(',')[0] not in data.selectionGroupNames():
            self.replot()
            return

        fdf = self.settingsWidget.calibration.fractionation(channel)

        if len(fdf) == 0:
            self.msg.show()
            self.replot()
            return

        self.msg.hide()
        grad = QCPColorGradient('gpViridis')
        externalsInUse = list(set(list(itertools.chain(*[c.property('External standard').split(',') for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]))))

        for i, intStd in enumerate(fdf['IS'].unique()):
            for ei, ext in enumerate(externalsInUse):
                t, r, rsd, fit = self.settingsWidget.calibration.fitFractionation(channel, intStd, group=ext)
                if t is None:
                    continue

                color = grad.color(i+ei+1, 1, len(fdf['IS'].unique())+len(externalsInUse))
                g = self.addGraph()
                g.setData(t, r)
                g.setLineStyle('lsNone')
                g.setScatterStyle('ssDisc', 6, color, color)
                g.setColor(color)
                name = (intStd[:20]+'...') if len(intStd) > 20 else intStd
                g.setName(f'{name} - {ext}')

                self.eb = QCPErrorBars(self.bottom(), self.left())
                self.eb.setDataPlottable(g)
                self.eb.setData(rsd)

                self.eb.removeFromLegend()

                if fit:
                    sg = self.addGraph()
                    sx = np.linspace(t.min(), t.max(), 100)
                    sg.setData(sx, fit(sx))
                    sg.setColor(color)
                    sg.removeFromLegend()

#        try:
#            sel = self.settingsWidget.sels[0]
#            num = data.timeSeries(channel+'_CPS').dataForSelection(sel)
#            den = data.timeSeries(sel.property('Internal element')+'_CPS').dataForSelection(sel)
#            r = savgol_filter(num, 11, 3)/savgol_filter(den, 11, 3)
#            g2 = self.addGraph(self.bottom(), self.right())
#            t = data.timeSeries(channel+'_CPS').timeForSelection(sel)
#            t -= t[0]
#            g2.setData(t, r)
#            print(r)

#        except:
#            print('No selection for fractionation plot')


        self.left().label = f'Normalized {data.timeSeries(channel).property("Element")}/Internal standard'
        self.rescaleAxes()
        self.replot()


class JacksonPlot(Plot):

    def __init__(self, parent):
        super().__init__(parent)
        self.settingsWidget = parent
        self.setToolsVisible(False)
        self.setBackground(CUI().tabBackgroundColor())
        self.bottom().label = 'Condensation temperature (K)'
        self.grad = QCPColorGradient('gpViridis')
        self.setLegendVisible(True)

    def updatePlot(self):
        self.clearGraphs()
        self.clearItems()

        if not hasattr(self.settingsWidget, 'sels') or not self.settingsWidget.sels:
            print('Did not update JacksonPlot because no selections...')
            return

        try:
            sel = self.settingsWidget.sels[0]
            isElements = sel.property('Internal element').split(',')

            channelName = self.settingsWidget.selectedChannelNames[0]
            self.left().label = f'{channelName} concentration (ppm)'
        except:
            return

        if not isElements or not isElements[0]:
            return

        thistc = data.elements[data.timeSeries(channelName).property('Element')]['Tcond_Lodders']
        tc = np.array([data.elements[data.timeSeries(ise).property('Element')]['Tcond_Lodders'] for ise in isElements])
        #tc = (thistc - tc)/tc

        cps = data.timeSeries(f'{channelName}_CPS')
        channelSurface = self.settingsWidget.calibration.surface(channelName, inv=True)

        conc = np.empty(len(tc))

        use_fg = drs.setting('UseFG')

        for i, ise in enumerate(isElements):
            try:
                isChannel = data.timeSeries(f'{ise}_CPS')
                isSurface = self.settingsWidget.calibration.surface(ise, inv=True)
                sqPPM = channelSurface(sel.midTimestamp, data.result(sel, cps).value())
                rmPPM = data.referenceMaterialData(sel.group().name)[data.timeSeries(ise).property('Element')].valueInUnits('fg') if use_fg else data.referenceMaterialData(sel.group().name)[data.timeSeries(ise).property('Element')].valueInPPM()
                isPPM = isSurface(sel.midTimestamp, data.result(sel, isChannel).value())
                conc[i] = rmPPM*sqPPM/isPPM
            except Exception as e:
                print(e)

        for i, ise in enumerate(isElements):
            g = self.addGraph()
            g.setData([tc[i]], [conc[i]])
            g.setLineStyle('lsNone')
            color = self.grad.color(i, 0, len(conc))
            g.setScatterStyle('ssDisc', 6, color, color)
            g.name = isElements[i]

        self.addStraightLine([thistc, conc.min()], [thistc, conc.max()])

        self.rescaleAxes()
        self.bottom().range = QCPRange(500, 1800)
        self.left().scaleRange(1.5)
        self.replot()

class ExtModelDialog(QDialog):

    def __init__(self, calibration, selectedChannels, parent=None):
        QDialog.__init__(self, parent)
        self.calibration = calibration
        self.calibration.updateBlocks()
        self.selectedChannels = selectedChannels
        self.blockSplines = {}
        try:
            self.setWindowTitle(f"Sensitivity model for {', '.join([ch.name for ch in self.selectedChannels])}")
        except:
            pass

        self.setLayout(QVBoxLayout())
        mainLayout = QHBoxLayout()
        leftWidget = QWidget(self)
        leftWidget.setLayout(QFormLayout())
        leftWidget.setFixedWidth(225)
        leftWidget.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addLayout(mainLayout)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        bb.accepted.connect(self.checkAndAccept)  #Can't exit Dialog without setting abundance.
        bb.rejected.connect(self.reject)

        self.layout().addWidget(bb)
        self.channelsTable = QTableWidget(self)
        self.allChannels = [c.name for c in data.timeSeriesList(data.Input) if 'Total' not in c.name]
        self.ion_energies = [data.elements[c.property("Element")]['ionizationEnergies'][0] for c in data.timeSeriesList(data.Input) if 'Total' not in c.name]
        self.channelsTable.setRowCount(len(self.allChannels))
        self.channelsTable.setColumnCount(2)
        self.channelsTable.setHorizontalHeaderLabels(['Channel', 'Abundance'])
        self.channelsTable.horizontalHeader().setStretchLastSection(True)
        self.channelsTable.verticalHeader().setVisible(False)
        self.channelsTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.channelsTable.setSelectionMode(QAbstractItemView.ExtendedSelection)

        for ri, name in enumerate(self.allChannels):
            self.channelsTable.setItem(ri, 0, QTableWidgetItem(name))
            self.channelsTable.item(ri, 0).setFlags(self.channelsTable.item(ri, 0).flags() & ~Qt.ItemIsEditable)
            element = data.timeSeries(name).property('Element')
            try:
                mass = int(data.timeSeries(name).property('Mass'))
                ab = [i for i in data.elements[element]['isotopes'] if i['massNumber'] == mass][0]['abundance']
            except Exception as e:
                ab = 1
            self.channelsTable.setItem(ri, 1, QTableWidgetItem(f'{ab:.6f}'))
            self.channelsTable.item(ri, 0).setData(Qt.UserRole, True)

            if data.timeSeries(name).property('External standard') == 'Model':
                # If this is a channel using a model, highlight the items differently
                self.channelsTable.item(ri, 0).setData(Qt.UserRole, False)
                self.channelsTable.item(ri, 0).setBackground(Qt.black)
                self.channelsTable.item(ri, 1).setBackground(Qt.black)
                self.channelsTable.item(ri, 0).setFlags(self.channelsTable.item(ri, 0).flags() & ~Qt.ItemIsSelectable)
                self.channelsTable.item(ri, 1).setFlags(self.channelsTable.item(ri, 1).flags() & ~Qt.ItemIsSelectable)

            if 'ModelChannels' in self.selectedChannels[0].properties():
                if name in self.selectedChannels[0].property('ModelChannels'):
                    index = self.channelsTable.model().index(ri, 0)
                    self.channelsTable.selectionModel().select(index, QItemSelectionModel.Rows | QItemSelectionModel.Select)

        self.plot = Plot(self)
        self.plot.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        leftWidget.layout().addRow(self.channelsTable)
        self.fitComboBox = QComboBox(self)
        self.fitComboBox.addItems([
            'MeanMean',
            'MeanMedian',
            'LinearFit',
            'WeightedLinearFit',
            'StepLinear',
            'StepForward',
            'StepBackward',
            'StepAverage',
            'Nearest',
            'Akima',
            'Spline_NoSmoothing',
            'Spline_Smooth1',
            'Spline_Smooth2',
            'Spline_Smooth3',
            'Spline_Smooth4',
            'Spline_Smooth5',
            'Spline_Smooth6',
            'Spline_Smooth7',
            'Spline_Smooth8',
            'Spline_Smooth9',
            'Spline_Smooth10',
            'Spline_AutoSmooth'
        ])
        self.fitComboBox.setCurrentText(self.fitType())
        self.fitComboBox.textActivated.connect(self.updateFitType)
        leftWidget.layout().addRow('Fit type', self.fitComboBox)

        self.ie_checkBox = QCheckBox('', self)
        self.ie_checkBox.setChecked(drs.setting('UseIonizationEnergies'))
        self.ie_checkBox.clicked.connect(self.toggle_ie)
        leftWidget.layout().addRow('Use Ionisation Energies?', self.ie_checkBox)

        mainLayout.addWidget(leftWidget)
        mainLayout.addWidget(self.plot)
        self.resize(1200, 600)

        self.plot.left().label = 'Abundance corrected CPS/ppm'
        self.plot.bottom().label = 'Channel (m/z)'
        self.colorGrad = QCPColorGradient('gpSpectrum')

        self.channelsTable.selectionModel().selectionChanged.connect(self.processSelectionChanged)
        self.channelsTable.cellChanged.connect(self.processCellChanged)

        self.showLegend = False

        menu = self.plot.contextMenu()
        menu.addSeparator()
        la = menu.addAction('Legend')
        la.setCheckable(True)
        la.setChecked(self.showLegend)
        la.triggered.connect(self.toggleLegend)

        self.updatePlot()


    def checkAndAccept(self):
        #Check that selected channels have an abudance set:
        for ch in self.selectedChannels:
            item = self.channelsTable.item(
                self.channelsTable.findItems(ch.name, Qt.MatchExactly)[0].row(), 1)
            if float(item.text()) < 0.000001:
                self.abunNotice = OverlayMessage(self.plot,
                                                 'Error',
                                                 f'Please set the isotopic abundance of {ch.name} before closing this dialog',
                                                 3, 0.8, 40, 50)
                self.abunNotice.setCloseButtonVisible(False)
                self.abunNotice.show()
                return

        self.accept()

    def toggleLegend(self, b):
        self.showLegend = b
        # Get axis ranges so that toggling the legend doesn't reset zoom
        l_range = self.plot.left().range
        b_range = self.plot.bottom().range
        self.updatePlot()
        self.plot.left().setRange(l_range)
        self.plot.bottom().setRange(b_range)
        self.plot.replot()

    def fitType(self):
        try:
            fitType = [ch.property('ModelFitType') for ch in self.selectedChannels][0]
            if not fitType:
                fitType = 'LinearFit'
        except:
            fitType = 'LinearFit'

        return fitType

    def updateFitType(self, fit_name):
        for ch in self.selectedChannels:
            ch.setProperty('ModelFitType', fit_name)

        self.updatePlot()

    def toggle_ie(self, b):
        drs.setSetting('UseIonizationEnergies', b)

        self.updatePlot()

    def processCellChanged(self, row, col):
        # Only process changes in col == 1 == abundance
        if col != 1:
            return

        channel_name = self.channelsTable.item(row, 0).text()
        print(f'Abundance changed for {channel_name}')

    def processSelectionChanged(self):
        channel_names = [index.data() for index in self.channelsTable.selectionModel().selectedRows()]
        for ch in self.selectedChannels:
            ch.setProperty('ModelChannels', channel_names)
        self.updatePlot()

    def updatePlot(self):
        print('Updating plot...')
        self.plot.clearGraphs()
        self.plot.clearItems()

        channelsForFit = [index.data() for index in self.channelsTable.selectionModel().selectedRows()]
        iForFit = [index.row() for index in self.channelsTable.selectionModel().selectedRows() if data.timeSeries(index.data()).property('External standard') != 'Model']
        iForFit.sort()

        channelMasses = np.array([int(data.timeSeries(n).property('Mass')) for n in self.allChannels], dtype=np.float)
        xOut = np.linspace(channelMasses.min(), channelMasses.max(), 1000)
        channelAbundances = np.array([float(self.channelsTable.item(ri, 1).text()) for ri in range(self.channelsTable.rowCount)])

        def sensForChannel(block, channel_name):
            if data.timeSeries(channel_name).property('External standard') == 'Model':
                return np.nan
            return block.slope(channel_name)

        # Quick note for future selves: can't plot channel sensitivities in one go because it will have to update if the user
        # changes any of the isotopic abundances in the table. So, can't put main plotting in __init__() for example
        # However, we can do it for ionization energies because the user can't change those
        for bi, block in enumerate(self.calibration.blocks):
            block_graph = self.plot.addGraph()
            block_graph.setName(f"Block {bi + 1}")
            if drs.setting('UseIonizationEnergies'):
                channelSens = np.array([sensForChannel(block, cn) for cn in self.allChannels], dtype=np.float) / channelAbundances / self.ion_energies
            else:
                channelSens = np.array([sensForChannel(block, cn) for cn in self.allChannels], dtype=np.float) / channelAbundances
            block_graph.setData(channelMasses[np.isfinite(channelSens)], channelSens[np.isfinite(channelSens)])
            color = self.colorGrad.color(bi, 0, len(self.calibration.blocks))
            block_graph.setScatterStyle('ssDisc', 6, color, color)
            block_pen = QPen(QColor(color))
            block_pen.setStyle(Qt.DotLine)
            block_graph.setPen(block_pen)

            xForFit = channelMasses[iForFit]
            yForFit = channelSens[iForFit]

            if len(xForFit) < 1:
                continue

            if len(xForFit) >= 1:
                yOut = data.spline(xForFit, yForFit, np.ones(len(yForFit)), self.fitType(), xOut)
                self.blockSplines[block.label] = (xOut, yOut)
                fitGraph = self.plot.addGraph()
                fitGraph.pen = QPen(color, 2)
                fitGraph.setData(xOut, yOut)
                fitGraph.removeFromLegend()

            # Also, highlight the masses used in the fit on the plot
            used_graph = self.plot.addGraph()
            used_graph.setData(xForFit, yForFit)
            used_graph.setScatterStyle('ssCircle', 10, Qt.red, Qt.transparent)
            used_graph.setLineStyle('lsNone')
            used_graph.removeFromLegend()


        for ch in self.selectedChannels:
            try:
                this_mass = float(ch.property('Mass'))
                y_Max = self.plot.left().range.upper()
                mass_line = self.plot.addStraightLine([this_mass,0], [this_mass, y_Max])
                mass_line.pen = QPen(Qt.DashLine)
            except Exception:
                # TODO: get mass from channel name???
                pass

        self.plot.rescaleAxes()
        # Going to get x maximum from channels because if there are a lot of channels in the actinides that are being
        # fitted, they won't be in the table. So get it from the dataManager
        try:
            last_ch_mass = channelMasses.max()
            self.plot.bottom().range = QCPRange(0, last_ch_mass + last_ch_mass * 0.1)
        except Exception:
            self.plot.bottom().range = QCPRange(0, self.plot.bottom().range.upper() + self.plot.bottom().range.upper() * 0.1)

        #self.plot.left().range = QCPRange(0, y_currentMax + y_currentMax * 0.1)

        self.plot.setLegendVisible(self.showLegend)
        self.plot.setLegendAlignment(Qt.AlignTop | Qt.AlignLeft)
        if drs.setting('UseIonizationEnergies'):
            self.plot.left().label = 'Abundance and ionization energy corrected CPS/ppm'
        else:
            self.plot.left().label = 'Abundance corrected CPS/ppm'


        self.plot.replot()

    def blockSensitivities(self, channel_name):
        from scipy.interpolate import interp1d
        # Need to get the abundance for this
        try:
            mass = float(data.timeSeries(channel_name).property('Mass'))
            table_row = self.channelsTable.findItems(channel_name, Qt.MatchFixedString)[0].row()
            print(f'Table row: {table_row}')
            abund = float(self.channelsTable.item(table_row, 1).text())
        except:
            print(f'Could not calculate block sensitivities for {channel_name}')
            mass = np.nan
            abund = np.nan
            ie = np.nan

        bs = {}
        for bi, block in enumerate(self.calibration.blocks):
            x, y = self.blockSplines[block.label]
            f = interp1d(x, y, 'linear', fill_value='extrapolate')
            S = f(mass)*abund
            bs[block.label] = S

        return bs


class DataStatus(Flag):
    Ready = 0
    NoBaselineSelections = auto()
    NoRMSelections = auto()
    NoInputs = auto()
    NoCPS = auto()
    NoInternalStandard = auto()
    NoExternalStandard = auto()
    MissingChannelMetadata = auto()
    MixedRMType = auto()


class SettingsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calibration = Calibration()

        settings = QSettings()
        self.ui_path = settings.value("Paths/DataReductionSchemesPath")
        self.ui_file = QFile(self.ui_path + "/3d_trace_elements.ui")

        self.setLayout(QVBoxLayout())
        if not self.ui_file.open(QIODevice.ReadOnly):
            raise RuntimeError('Could not load settings ui')

        ui = QUiLoader().load(self.ui_file, self)
        self.layout().addWidget(ui)
        self.layout().setContentsMargins(0, 0, 0, 0)

        self.extTable = ui.findChild(QTableView, 'externalTable')
        self.extFilter = ui.findChild(QLineEdit, 'filterES')
        self.intGroupBox = ui.findChild(QGroupBox, 'ISGroupBox')
        self.intTable = ui.findChild(QTableView, 'internalTable')
        self.intFilter = ui.findChild(QLineEdit, 'filterIS')
        self.tabWidget = ui.findChild(QTabWidget, 'tabWidget')
        self.setExtButton = ui.findChild(QToolButton, 'setExternalButton')
        self.setIntButton = ui.findChild(QToolButton, 'internalElementButton')
        self.setIntValueButton = ui.findChild(QToolButton, 'internalValueButton')
        self.setIntUnitsButton = ui.findChild(QToolButton, 'internalUnitsButton')
        self.importValuesButton = ui.findChild(QToolButton, 'importValuesButton')
        self.setIndexChannelCBox = ui.findChild(QComboBox, 'indexChannel')
        self.maskGroupBox = ui.findChild(QGroupBox, 'maskGroupBox')
        self.maskChannelCBox = ui.findChild(QComboBox, "maskChannel")
        self.maskMethodCBox = ui.findChild(QComboBox, "maskMethodComboBox")
        self.maskLineEdit = ui.findChild(QLineEdit, "maskCutoff")
        self.maskTrimSB = ui.findChild(QDoubleSpinBox, "maskTrim")
        self.maskFrame = ui.findChild(QFrame, 'maskFrame')
        self.normalizeExtCheckBox = ui.findChild(QCheckBox, 'normalizeExtCheckBox')
        self.fitMethodComboBox = ui.findChild(QComboBox, 'fitMethodComboBox')
        self.externalSplitter = ui.findChild(QSplitter, 'externalSplitter')
        self.normToComboBox = ui.findChild(QComboBox, 'normToComboBox')
        self.splineTypeComboBox = ui.findChild(QComboBox, 'splineTypeComboBox')
        self.splineTypeComboBox.addItems([
            'MeanMean',
            'MeanMedian',
            'LinearFit',
            'WeightedLinearFit',
            'StepLinear',
            'StepForward',
            'StepBackward',
            'StepAverage',
            'Nearest',
            'Akima',
            'Spline_NoSmoothing',
            'Spline_Smooth1',
            'Spline_Smooth2',
            'Spline_Smooth3',
            'Spline_Smooth4',
            'Spline_Smooth5',
            'Spline_Smooth6',
            'Spline_Smooth7',
            'Spline_Smooth8',
            'Spline_Smooth9',
            'Spline_Smooth10',
            'Spline_AutoSmooth'
        ])
        self.statComboBox = ui.findChild(QComboBox, 'statComboBox')
        self.fgButton = ui.findChild(QToolButton, 'fgButton')
        self.setupGroupPropsButton = ui.findChild(QToolButton, 'setupGroupPropsButton')
        self.criteriaButton = ui.findChild(QToolButton, 'criteriaButton')
        self.throughZeroButton = ui.findChild(QToolButton, 'throughZeroButton')
        self.fracButton = ui.findChild(QToolButton, 'fracToolButton')
        self.modelComboBox = ui.findChild(QComboBox, 'modelComboBox')
        self.bsMethodComboBox = ui.findChild(QComboBox, 'bsMethodComboBox')
        self.bsChannelComboBox = ui.findChild(QComboBox, 'bsChannelComboBox')
        self.bsLineEdit = ui.findChild(QLineEdit, 'bsLineEdit')
        self.bsFrame = ui.findChild(QFrame, 'bsFrame')
        self.affinityButton = ui.findChild(QToolButton, 'affinityButton')
        self.affinityCorrectionCheckBox = ui.findChild(QCheckBox, 'affinityCorrectionCheckBox')
        self.affinityCorrectionSpinBox = ui.findChild(QDoubleSpinBox, 'affinityCorrectionSpinBox')

        self.throughZeroButton.setIcon(CUI().icon('bullseye'))
        self.throughZeroButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.throughZeroButton.setToolTip("Force calibration curve through 0")

        self.fracButton.setIcon(CUI().icon('magic'))
        self.fracMenu = QMenu(self)
        fracActions = [self.fracMenu.addAction(t) for t in ['None', 'Linear', 'Spline']]
        self.fracButton.setMenu(self.fracMenu)
        self.fracButton.setPopupMode(QToolButton.DelayedPopup)

        self.popoutButton = QToolButton(self.tabWidget)
        self.popoutButton.setIcon(CUI().icon('externallink'))
        self.popoutButton.setCheckable(True)
        self.popoutButton.setChecked(False)
        self.popoutButton.clicked.connect(self.togglePlotEmbedding)
        self.tabWidget.installEventFilter(self)
        self.tabWidget.setCornerWidget(self.popoutButton)
        self.tabWidget.setWindowTitle('3D Trace Elements')
        self.tabWidget.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setupGroupPropsButton.setIcon(CUI().icon('cog'))
        timeSeriesNames = data.timeSeriesNames(data.Input)
        defaultChannelName = ""
        if timeSeriesNames:
            defaultChannelName = timeSeriesNames[0]

        drs.setDefaultSetting("IndexChannel", defaultChannelName)
        drs.setDefaultSetting("Mask", False)
        drs.setDefaultSetting("MaskChannel", defaultChannelName)
        drs.setDefaultSetting('MaskMethod', 'Laser log')
        drs.setDefaultSetting("MaskCutoff", 0.1)
        drs.setDefaultSetting("MaskTrim", 0.0)
        drs.setDefaultSetting("NormalizeExternals", True)
        drs.setDefaultSetting('StatName', 'mean')
        drs.setDefaultSetting('UseFG', False)
        drs.setDefaultSetting("UseIntStds", True)
        drs.setDefaultSetting('SplineType', 'Spline_AutoSmooth')
        drs.setDefaultSetting('MasterExternal', '')
        bsDefault = 'Laser log' if len(data.laserLogSamples()) > 0 else 'Cutoff threshold'
        drs.setDefaultSetting('BeamSecondsMethod', bsDefault)
        drs.setDefaultSetting('BeamSecondsChannel', defaultChannelName)
        drs.setDefaultSetting('BeamSecondsValue', 1000.)
        drs.setDefaultSetting('AffinityCorrection', False)
        drs.setDefaultSetting('AffinityCorrection%', 15.0)
        drs.setDefaultSetting('BlockFindingMethod', 'Simple')
        drs.setDefaultSetting('NClusters', -1)
        drs.setDefaultSetting('ISCriteria', [])
        drs.setDefaultSetting('UseIonizationEnergies', False)

        useIsotopicConcentrationsFromPrefs = settings.value('UseIsotopicConcentrations', False)
        if isinstance(useIsotopicConcentrationsFromPrefs, str):
            drs.setSetting('UseIsotopicConcentrations', useIsotopicConcentrationsFromPrefs.lower() == 'true')
        else:
            drs.setSetting('UseIsotopicConcentrations', useIsotopicConcentrationsFromPrefs)

        self.maskTrimSB.setMinimum(-1E5)
        self.maskTrimSB.setMaximum(1E5)

        self.setupChannelCBs()
        self.setupExt()
        self.setupInt()
        self.setup3dPlot()
        self.setupBlockPlot()
        self.setupFitsPlot()
        self.setupFitParamsPlot()
        self.setupFractionationPlot()

        self.blockRMs = []

        # Restore settings
        print(f'Restoring settings {drs.settings()}')
        self.normalizeExtCheckBox.setChecked(drs.setting('NormalizeExternals'))
        self.setIndexChannelCBox.currentText = drs.setting('IndexChannel')
        self.maskLineEdit.setText(str(drs.setting('MaskCutoff')))
        self.maskTrimSB.setValue(float(drs.setting('MaskTrim')))
        self.maskChannelCBox.currentText = drs.setting('MaskChannel')
        self.maskMethodCBox.currentText = drs.setting('MaskMethod')
        self.maskFrame.setVisible('log' not in drs.setting('MaskMethod'))
        self.intGroupBox.setChecked(bool(drs.setting('UseIntStds')))
        self.splineTypeComboBox.currentText = drs.setting('SplineType')
        self.maskGroupBox.setChecked(drs.setting('Mask'))
        self.normToComboBox.currentText = drs.setting('MasterExternal')
        self.bsMethodComboBox.currentText = drs.setting('BeamSecondsMethod')
        self.bsChannelComboBox.currentText = drs.setting('BeamSecondsChannel')
        self.bsLineEdit.setText(drs.setting('BeamSecondsValue'))
        self.bsFrame.setVisible('log' not in drs.setting('BeamSecondsMethod').lower())
        self.affinityCorrectionCheckBox.setChecked(drs.setting('AffinityCorrection'))
        self.affinityCorrectionSpinBox.setValue(drs.setting('AffinityCorrection%'))
        self.statComboBox.currentText = drs.setting('StatName')
        self.fgButton.setChecked(drs.setting('UseFG'))
        self.setupGroupPropsButton.setVisible(drs.setting('UseFG'))

        self.sels = []

        # Connections
        self.setIndexChannelCBox.activated.connect(lambda t: drs.setSetting("IndexChannel", self.setIndexChannelCBox.currentText))
        self.maskGroupBox.toggled.connect(lambda b: drs.setSetting('Mask', b))
        self.maskMethodCBox.activated.connect(lambda t: drs.setSetting('MaskMethod', self.maskMethodCBox.currentText))
        self.maskMethodCBox.activated.connect(lambda t: self.maskFrame.setVisible(t))
        self.maskChannelCBox.activated.connect(lambda t: drs.setSetting("MaskChannel", self.maskChannelCBox.currentText))
        self.maskLineEdit.textEdited.connect(lambda t: drs.setSetting("MaskCutoff", float(t)))
        self.maskTrimSB.valueChanged.connect(lambda t: drs.setSetting("MaskTrim", float(t)))
        self.normalizeExtCheckBox.toggled.connect(lambda b: self.updateDRSSetting('NormalizeExternals', b))
        self.normToComboBox.activated.connect(lambda t: self.updateDRSSetting('MasterExternal', self.normToComboBox.currentText))
        self.splineTypeComboBox.activated.connect(lambda t: self.updateDRSSetting('SplineType', self.splineTypeComboBox.currentText))
        self.intGroupBox.toggled.connect(lambda b: drs.setSetting('UseIntStds', b))
        self.criteriaButton.clicked.connect(self.editCriteria)
        self.throughZeroButton.clicked.connect(self.toggleThroughZero)
        self.importValuesButton.clicked.connect(self.importValues)
        self.fracButton.clicked.connect(self.toggleFractionation)
        self.fracMenu.triggered.connect(self.toggleFractionation)
        data.dataChanged.connect(self.setupChannelCBs)
        self.modelComboBox.activated.connect(self.changeModel)
        self.bsMethodComboBox.activated.connect(lambda t: drs.setSetting('BeamSecondsMethod', self.bsMethodComboBox.currentText))
        self.bsMethodComboBox.activated.connect(lambda t: self.bsFrame.setVisible(t in [1, 2]))
        self.bsChannelComboBox.activated.connect(lambda t: drs.setSetting('BeamSecondsChannel', self.bsChannelComboBox.currentText))
        self.bsLineEdit.textEdited.connect(lambda t: drs.setSetting('BeamSecondsValue', float(t)))
        drs.finished.connect(lambda: drs.setProperty('isRunning', False))

        drs.finished.connect(lambda: self.intModel.refreshSelections())

        self.intTable.installEventFilter(self)
        self.statComboBox.activated.connect(lambda t: self.updateDRSSetting('StatName', self.statComboBox.currentText))
        self.fgButton.clicked.connect(lambda b: self.setupGroupPropsButton.setVisible(b))
        self.fgButton.clicked.connect(lambda b: self.updateDRSSetting('UseFG', b))
        self.setupGroupPropsButton.clicked.connect(self.setupGroupProps)

        # Create notices that may or may not be hidden
        self.extNotice = OverlayMessage(self.extTable, 'Warning', 'Baseline subtracted channels are missing.', 0, 0.8, 40, 10)
        self.extNotice.setCloseButtonVisible(False)
        self.blsButton = QToolButton(self.extNotice)
        self.blsButton.setText('Calculate now!')
        self.blsButton.clicked.connect(self.extModel.refreshChannels)
        self.extNotice.addButton(self.blsButton)
        self.extNotice.hide()

        self.noSelsNotice = OverlayMessage(self.extTable, 'Warning', 'There are no selections for one of the External Standards',
                                           0, 0.8, 40, 10)
        self.noSelsNotice.setCloseButtonVisible(False)
        self.noSelsNotice.hide()

        self.intNotice = OverlayMessage(self.intTable, 'Warning', 'Elements and values must be specified for > 0 selections.', 0, 0.8, 40, 10)
        self.intNotice.setCloseButtonVisible(False)
        self.intNotice.hide()

        self.mdNotice = OverlayMessage(self.extTable, 'Warning', 'There is missing metadata for some channels.', 0, 0.8, 40, 10)
        self.mdNotice.setCloseButtonVisible(False)
        self.mdNotice.hide()

        self.updateStatus()

    def setupGroupProps(self):
        from iolite.ui import DynamicPropertyEditor, DynamicPropertyModel
        d = QDialog()
        d.setLayout(QVBoxLayout())
        toolTips = ['Selection group name',
                    'Thickness of sample in μm',
                    'Density of sample in g.cm⁻³',
                    'Spot size in μm',
                    'Spot shape (either \'rect\' or \'circle\')']
        gp_model = DynamicPropertyModel(data.selectionGroupList(data.ReferenceMaterial | data.Sample), ['Name', 'Thickness', 'Density', 'SpotSize', 'SpotShape'], toolTips)
        gp_model.disableEditingForProperty('Name')
        gp_view = DynamicPropertyEditor()
        gp_view.setModel(gp_model)
        gp_view.setAttribute(Qt.WA_DeleteOnClose)
        d.layout().addWidget(gp_view)

        bb = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.RestoreDefaults, d)
        d.layout().addWidget(bb)
        bb.rejected.connect(d.reject)
        bb.accepted.connect(d.accept)

        bb.button(QDialogButtonBox.RestoreDefaults).setText('Set selected')

        def set_selected():
            v = QInputDialog.getText(d, 'Group property value', 'New value')
            if not v:
                return

            for index in gp_view.selectedIndexes():
                try:
                    gp_model.setData(index, float(v))
                except:
                    gp_model.setData(index, v)

        bb.button(QDialogButtonBox.RestoreDefaults).clicked.connect(set_selected)
        d.resize(600, 500)
        d.exec()



    def updateStatus(self):
        self.status = DataStatus.Ready
        if len(data.timeSeriesNames(data.Input)) == 0:
            self.status = self.status | DataStatus.NoInputs
        if len(data.selectionGroupList(data.Baseline)) == 0 or len(data.selectionGroupList(data.Baseline)[0].selections()) == 0:
            self.status = self.status | DataStatus.NoBaselineSelections
        if len(data.timeSeriesNames(data.Intermediate, {'DRSType': 'BaselineSubtracted'})) == 0:
            self.status = self.status | DataStatus.NoCPS
        if len(data.selectionGroupList(data.ReferenceMaterial)) == 0 or len(data.selectionGroupList(data.ReferenceMaterial)[0].selections()) == 0:
            self.status = self.status | DataStatus.NoRMSelections
        if len(self.externalsInUse()) == 0:
            self.status = self.status | DataStatus.NoExternalStandard
        if len(self.internalsInUse()) == 0:
            self.status = self.status | DataStatus.NoInternalStandard
        #print(f'Updated status to {self.status}')
        self.maybeShowExtNotices()


    def eventFilter(self, obj, event):
        if obj == self.tabWidget and event.type()== QEvent.Close:
            print(f'Caught close event on tabWidget')
            event.ignore()
            return True

        def is_float(v):
            try:
                float(v)
                return True
            except:
                return False

        if obj == self.intTable and event.type() == QEvent.KeyPress and event.matches(QKeySequence.Paste):
            text = QApplication.clipboard().text()
            values = re.split('\s+|,', text)
            values = [float(v) for v in values]# if is_float(v)]

            if len(values) > 1: # If multiple values pasted, use them up
                indCount = 0
                for ind in self.intTable.selectionModel().selectedRows(3):
                    if indCount >= len(values):
                        break

                    self.intModel.setData(ind, values[indCount])
                    indCount += 1
            else: # If only 1 value pasted, paste it to all selected rows
                for ind in self.intTable.selectionModel().selectedRows(3):
                    self.intModel.setData(ind, values[0])


        return False

    def setupChannelCBs(self):
        cbs = [self.setIndexChannelCBox, self.maskChannelCBox, self.bsChannelComboBox]

        for cb in cbs:
            ct = cb.currentText
            cb.clear()
            cb.addItems(data.timeSeriesNames(data.Input))

            if cb.findText(ct) >= 0:
                cb.currentText = ct

    def externalsInUse(self):
        groupNames = list(itertools.chain(*[str(self.extModel.index(r, 1).data()).split(',') for r in range(self.extModel.rowCount())]))
        groupNames = list(set(groupNames))
        try:
            groupNames.remove('Model')
            groupNames.remove('None')
        except:
            pass
        return groupNames

    def internalsInUse(self):
        isElements = [str(s.property('Internal element')) for s in self.intModel.selections]
        isElements = list(set(isElements))
        if 'None' in isElements:
            isElements.remove('None')
        if '' in isElements:
            isElements.remove('')
        return isElements


    def updateNormOptions(self):
        current = self.normToComboBox.currentText
        self.normToComboBox.clear()
        self.normToComboBox.addItems(self.externalsInUse())
        index = self.normToComboBox.findText(current)
        if index >= 0:
            self.normToComboBox.currentIndex = index

    def setupExt(self):
        self.extTable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.extTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.extFilterModel = QSortFilterProxyModel(self)
        self.extModel = ExternalsModel(self)
        self.extFilterModel.setSourceModel(self.extModel)
        self.extTable.setModel(self.extFilterModel)
        self.extFilter.textEdited.connect(self.extFilterModel.setFilterFixedString)
        self.extTable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.extTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.extTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.extTable.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

        self.extTable.setContextMenuPolicy(Qt.CustomContextMenu)
        self.extTable.customContextMenuRequested.connect(self.showExtContextMenu)

        self.rmMenu = ReferenceMaterialsMenu(self)
        self.setExtButton.setMenu(self.rmMenu)
        self.setExtButton.setPopupMode(QToolButton.InstantPopup)
        self.setExtButton.setIcon(CUI().icon('trophy'))
        self.setExtButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.extModel.dataChanged.connect(lambda a: self.updateAffected())

        self.extModel.dataChanged.connect(lambda a: self.updateBlocks())
        self.extTable.selectionModel().selectionChanged.connect(self.processExtSelection)
        self.rmMenu.rmsChanged.connect(self.extModel.updateData)
        self.extModel.dataChanged.connect(self.updateNormOptions)
        self.updateNormOptions()
        if 'MasterExternal' in drs.settings():
            self.normToComboBox.currentText = drs.setting('MasterExternal')

        self.extModel.dataChanged.connect(self.processExtSelection)
        self.extTable.setItemDelegate(ExternalsDelegate(self.extTable))
        #data.dataChanged.connect(lambda: self.extModel.refreshChannels())
        data.dataChanged.connect(self.updateStatus)

    def showExtContextMenu(self, point):
        index = self.extTable.indexAt(point)
        rmsIndex = self.extModel.index(index.row(), 1)
        rms = str(rmsIndex.data())
        if rms != 'Model':
            print('This channel does not use a model, so not showing menu...')
            return
        menu = QMenu()
        action = menu.addAction('Configure Model')
        action.triggered.connect(self.configureExtModel)
        menu.exec(self.extTable.mapToGlobal(point))

    def configureExtModel(self):
        try:
            w = ExtModelDialog(self.calibration, self.selectedChannels)
        except MissingRMGroupError:
            self.status = DataStatus.NoRMSelections
            self.maybeShowExtNotices()
            return

        if w.exec() != QDialog.Accepted:
            return

        # Apply dialog settings to selected channels...
        for channel in self.selectedChannels:
            bs = w.blockSensitivities(channel.name)

            if drs.setting('UseIonizationEnergies') and len(bs.keys()) > 0:
                print("Correcting for ionization energies")
                i_e = data.elements[channel.property("Element")]['ionizationEnergies'][0]

                for key in bs.keys():
                    bs[key] = bs[key] * i_e

            else:
                print("No ionization energy correction...")

            channel.setProperty('ModelSensitivities', bs)

        self.processExtSelection()

    def maybeShowExtNotices(self):
        if drs.property('isRunning'):
            self.extNotice.hide()
            self.intNotice.hide()
            return

        if self.status & DataStatus.NoCPS:
            self.extNotice.show()
        else:
            self.extNotice.hide()

        if self.status & DataStatus.NoRMSelections:
            self.noSelsNotice.show()
        else:
            self.noSelsNotice.hide()

        if self.status & DataStatus.NoInternalStandard and drs.setting('UseIntStds'):
            self.intNotice.show()
        else:
            self.intNotice.hide()

        if self.status & DataStatus.MissingChannelMetadata:
            self.mdNotice.show()
        else:
            self.mdNotice.hide()

    def setupInt(self):
        self.intTable.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.intTable.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.intFilterModel = QSortFilterProxyModel(self)
        self.intModel = InternalsModel(self)
        self.intFilterModel.setSourceModel(self.intModel)
        self.intTable.setModel(self.intFilterModel)
        self.intFilter.textEdited.connect(self.intFilterModel.setFilterFixedString)
        self.intTable.selectionModel().selectionChanged.connect(self.processIntSelection)

        self.elementMenu = ChannelsMenu(self)
        self.setIntButton.setMenu(self.elementMenu)
        self.setIntButton.setPopupMode(QToolButton.InstantPopup)

        self.setIntValueButton.clicked.connect(self.getInternalValue)
        self.elementMenu.channelsChanged.connect(self.intModel.updateData)
        self.elementMenu.channelsChanged.connect(lambda: self.calibration.clearFractionationCache())
        self.elementMenu.channelsChanged.connect(lambda: self.fracPlot.updatePlot())
        self.elementMenu.channelsChanged.connect(self.updateStatus)

        self.affinityMenu = ChannelsMenu(self, propName = 'Affinity elements')
        self.affinityMenu.removeAction(self.affinityMenu.actions()[-1])
        majorsAction = self.affinityMenu.addAction('Majors')
        majorsAction.triggered.connect(self.setAffinitiesToMajors)
        self.affinityButton.setMenu(self.affinityMenu)
        self.affinityMenu.channelsChanged.connect(self.intModel.updateData)
        self.affinityButton.clicked.connect(lambda: self.intModel.refreshSelections())
        self.affinityCorrectionCheckBox.clicked.connect(lambda b: drs.setSetting('AffinityCorrection', b))
        self.affinityCorrectionSpinBox.valueChanged.connect(lambda v: drs.setSetting('AffinityCorrection%', v))

        affRMMenu = QMenu('Externals in use', self)
        self.affinityMenu.addMenu(affRMMenu)

        def updateAffRMMenu():
            print('Updating affinities menu...')
            affRMMenu.clear()
            for rm in self.externalsInUse():
                affRMMenu.addAction(rm)
        affRMMenu.aboutToShow.connect(updateAffRMMenu)
        affRMMenu.triggered.connect(lambda a: self.setAffinitiesToMaterial(a.text))

        self.unitsMenu = QMenu(self)
        for u in ['ppm', 'ppb', 'wtpc', 'wtpc_oxide']:
            self.unitsMenu.addAction(u)
        self.setIntUnitsButton.setMenu(self.unitsMenu)
        self.setIntUnitsButton.setPopupMode(QToolButton.InstantPopup)
        self.unitsMenu.triggered.connect(self.processUnits)
        self.intTable.setItemDelegate(InternalsDelegate(self.intTable))
        data.selectionGroupsChanged.connect(lambda: self.intModel.refreshSelections())
        data.selectionGroupsChanged.connect(self.updateStatus)

    def setAffinitiesToMajors(self):
        allMajors = ['Si', 'Al', 'Ca', 'Mg', 'Na', 'K', 'Ti', 'Fe', 'Mn', 'P']
        majors = [ch.name for ch in data.timeSeriesList(data.Input) if ch.property('Element') in allMajors]
        sels_to_update = self.sels

        if len(sels_to_update) < 1:     #If no selections selected in table, apply to all
            sels_to_update = self.intModel.selections

        for s in sels_to_update:
            s.setProperty('Affinity elements', ','.join(majors))

        self.intModel.updateData(self.sels)

    def setAffinitiesToMaterial(self, material):
        sels_to_update = self.sels

        if len(sels_to_update) < 1:     #If no selections selected in table, apply to all
            sels_to_update = self.intModel.selections

        for s in sels_to_update:
            s.setProperty('Affinity elements', material)

        self.intModel.updateData(self.sels)

    def setup3dPlot(self):
        self.plot3d = Plot3d(self)
        self.plot3d.setLabelFont(QFont('Helvetica', 16, QFont.Bold))
        self.tabWidget.addTab(self.plot3d, '3D')
        self.resetViewButton = OverlayButton(self.plot3d, 'TopLeft', 10, 10)
        self.resetViewButton.setIcon(CUI().icon('cube'))
        self.resetViewButton.setFixedSize(25, 25)
        self.resetViewButton.clicked.connect(self.reset3dView)
        self.reset3dView()

    def reset3dView(self):
        self.plot3d.setView(1, [1, 1, 1], [30, 0, 45])

    def togglePlotEmbedding(self, b):
        if b:
            # Out
            label = QLabel('Visualization in external window')
            label.setAlignment(Qt.AlignCenter)
            CUI().splitterReplace(self.externalSplitter, 1, label)
            self.tabWidget.show()
            self.tabWidget.resize(600, 600)
            try:
                self.update3d(self.selectedChannelNames[0])
            except:
                pass
        else:
            # In
            CUI().splitterReplace(self.externalSplitter, 1, self.tabWidget)
            try:
                self.update3d(self.selectedChannelNames[0])
            except:
                pass

    def setupBlockPlot(self):
        self.blockPlot = BlockPlot(self)
        self.tabWidget.addTab(self.blockPlot, 'Blocks')

    def setupFitsPlot(self):
        self.fitsPlot = FitsPlot(self)
        self.tabWidget.addTab(self.fitsPlot, 'Fits')

    def setupFitParamsPlot(self):
        self.fitParamsPlot = FitParamsPlot(self)
        self.tabWidget.addTab(self.fitParamsPlot, 'Fit Params')

    def setupFractionationPlot(self):
        self.fracPlot = FractionationPlot(self)
        self.tabWidget.addTab(self.fracPlot, 'Fractionation')

        #self.jackPlot = JacksonPlot(self)
        #self.tabWidget.addTab(self.jackPlot, 'Condensation T')

    def updateBlocks(self):
        if not drs.setting('MasterExternal') or drs.setting('MasterExternal') not in self.externalsInUse():
            print(f'Master external is invalid...')
            if self.externalsInUse() and len(self.externalsInUse()) > 0:
                me = self.externalsInUse()[0]
                drs.setSetting('MasterExternal', me)
                self.normToComboBox.currentText = me
                print(f'   ... setting it to {me}.')
            else:
                print('   ... because there are no externals in use.')

        if self.blockRMs != self.externalsInUse():
            try:
                self.calibration.updateBlocks()
            except MissingRMGroupError:
                return

            self.blockRMs = self.externalsInUse()

    def compileMeasurementData(self, channelName, ex, ey, ez):
        mx = np.array([]); my = np.array([]); mz = np.array([])
        msx = np.array([]); msy = np.array([]); msz = np.array([])
        grad = QCPColorGradient('gpViridis')
        colors = []

        for bi, block in enumerate(self.calibration.blocks):
            try:
                df = block.dataFrameForChannel(channelName)
                # Users may have created their own RM files, and put 0 in for missing values.
                # This will mess with calculations below, so remove those here
                df = df[df[f'{channelName}_RMppm'] != 0]
                mx = np.append(mx, df['sel_mid_time'])
                msx = np.append(msx, df['sel_duration'])
                my = np.append(my, df[f'{channelName}_RMppm'])
                msy = np.append(msy, df[f'{channelName}_RMppm_Uncert'])
                mz = np.append(mz, df[f'{channelName}'])
                msz = np.append(msz, df[f'{channelName}_Uncert'])
                c = grad.color(bi, 0, len(self.calibration.blocks))
                c.setAlpha(120)
                colors += [c]*len(df)
            except Exception:
                # Probably no data for selected channel for this block, so skip...
                continue

        # Normalize to the range being sent for surface
        msx = (msx/mx); msy = msy/my; msz = msz/mz
        mx = (mx-ex[0])/(ex[1] - ex[0])
        my = (my-ey[0])/(ey[1] - ey[0])
        mz = (mz-ez[0])/(ez[1] - ez[0])
        msx = msx*mx; msy = msy*my; msz = msz*mz
        return mx, msx, my, msy, mz, msz, colors

    def update3d(self, channelName, n=100):
        try:
            cpsChannel = data.timeSeries(f'{channelName}_CPS')
            minSlope = np.nanmin([abs(block.slope(channelName)) for block in self.calibration.blocks])
            minInt = np.nanmin([block.intercept(channelName) for block in self.calibration.blocks])
            # cps = slope*ppm + intercept
            af, _ = calculateRelativeYields()
            aff = 1.0
            if af is not None:
                aff = 1./np.nanmin(np.fromiter(af.values(), dtype=float)) # To compensate for low yield
            maxppm = aff * np.nanmax( (cpsChannel.data() - minInt) / (minSlope))
            # For some reason I haven't figured out yet, the max ppm might be less than the max ppm of
            # the highest ref mat. Check for that here...
            # Should only affect display
            rm_ppm = np.array([])
            rm_ppm_uncert = np.array([])
            for bi, block in enumerate(self.calibration.blocks):
                df = block.dataFrameForChannel(channelName)
                df = df[df[f'{channelName}_RMppm'] != 0]
                rm_ppm = np.append(rm_ppm, df[f'{channelName}_RMppm'])
                rm_ppm_uncert = np.append(rm_ppm_uncert, df[f'{channelName}_RMppm_Uncert'])

            if len(rm_ppm) > 1:
                max_rm_ppm = rm_ppm[np.nanargmax(rm_ppm)] + rm_ppm_uncert[np.nanargmax(rm_ppm)]
                maxppm = max(maxppm, max_rm_ppm)

            x = np.linspace(cpsChannel.time().min(), cpsChannel.time().max(), n)
            y = np.linspace(0, maxppm, n)
            X,Y = np.meshgrid(x,y)
            Z = self.surface(X,Y)
            Z = Z.astype(np.float32)
            self.plot3d.setSurfaceData(Z)
            self.plot3d.setXTitle('Time')
            if drs.setting('UseFG'):
                self.plot3d.setYTitle('Mass (fg)')
            else:
                self.plot3d.setYTitle('Concentration')
            self.plot3d.setZTitle('Intensity')
            for ax in ['X1', 'X2', 'X3', 'X4', 'Y1', 'Y2', 'Y3', 'Y4', 'Z1', 'Z2', 'Z3', 'Z4']:
                self.plot3d.setTickLabels(ax, [0, 1], ['Min', 'Max'])

            self.plot3d.setOrtho(True)
            self.plot3d.setMeasurementData(*self.compileMeasurementData(channelName, (x.min(), x.max()), (y.min(), y.max()), (Z.min(), Z.max())))
        except Exception as e:
            # if there is any problem getting the above data, clear the 3d plot
            # probably they have not set an external for the selected group
            print(f'Problem updating 3d plot: {e}')
            self.plot3d.clearData()

    def updateAffected(self):
        # If one of the channels modified is part of an IS we also need to
        # clear all the stored fractionation and surface caches
        selectedNames = [sr.data(Qt.UserRole).name for sr in self.extTable.selectionModel().selectedRows()]
        print(f'Update Affected {selectedNames}')
        if any(name in ''.join(self.internalsInUse()) for name in selectedNames):
            print('... clearing caches!')
            self.calibration.clearFractionationCache()
            self.calibration.clearSurfaceCache()

        [self.calibration.updateSurface(name) for name in selectedNames]

        try:
            [self.calibration.updateFractionation(name) for name in selectedNames]
        except RuntimeError:
            return


    def processExtSelection(self):
        self.selectedChannels = [sr.data(Qt.UserRole) for sr in self.extTable.selectionModel().selectedRows()]
        if not self.selectedChannels:
            self.selectedChannels = [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]

        self.selectedChannelNames = [c.name for c in self.selectedChannels]

        # Check that all channels have an element set so that it doesn't cause errors below
        for ch in data.timeSeriesList(data.Input):
            if ch.name == 'TotalBeam' or ch.name == 'AllLight' or ch.name == 'x [um]' or ch.name == 'y [um]':
                continue
            if not ch.property('Element') or len(ch.property('Element')) < 1:
                print(f"There was an issue with {ch.name}")
                self.status = self.status | DataStatus.MissingChannelMetadata
                self.maybeShowExtNotices()
                return

        self.rmMenu.rmsForActiveChannels = []
        for c in self.selectedChannels:
            if c.property('External standard') is not None:
                self.rmMenu.rmsForActiveChannels.append(c.property('External standard'))

        self.rmMenu.activeChannels = self.selectedChannelNames
        if len(self.selectedChannels) < 1:
            return

        channel = self.selectedChannels[0]
        fitThroughZero = bool(channel.property('FitThroughZero'))
        frac = bool(channel.property('FractionationCorrection'))
        self.throughZeroButton.setChecked(fitThroughZero)
        self.fracButton.setChecked(frac)
        model = channel.property('Model')
        if model:
            self.modelComboBox.setCurrentText(model)

        if len(self.calibration.blocks) == 0:
           try:
               self.updateBlocks()
           except RuntimeError:
               self.status = DataStatus.NoRMSelections
               return

        if len(self.calibration.blocks) == 0:
            print('No blocks. Aborting update!')
            return

        self.blockPlot.updatePlot()
        self.fitsPlot.updatePlot()
        self.fitParamsPlot.updatePlot()
        self.fracPlot.updatePlot()
        #self.jackPlot.updatePlot()

        # Check to make sure we have RM values for this channel
        extStd = channel.property('External standard')
        if f'{channel.name}_RMppm' not in self.calibration.blocks[0].dataFrame().columns:
            self.surface = None
        elif not self.calibration.blocks[0].dataFrame()[f'{channel.name}_RMppm'].notna().values.any() and extStd != 'Model':
            self.surface = None
        elif type(extStd) == str and (extStd.split(',')[0] in data.selectionGroupNames() or extStd == 'Model'):
            self.surface, _ = fitSurface(self.calibration.blocks, channel.name)
            # fitSurface may have changed the spline type, so update UI here:
            self.splineTypeComboBox.currentText = drs.setting('SplineType')
        else:
            self.surface = None

        self.update3d(channel.name)

    def processIntSelection(self):
        self.sels = [sr.data(Qt.UserRole) for sr in self.intTable.selectionModel().selectedRows()]
        if not self.sels:
            self.sels = self.intModel.selections

        self.elementMenu.setSelections(self.sels)
        self.affinityMenu.setSelections(self.sels)
        #self.jackPlot.updatePlot()

    def getInternalValue(self):
        d = QDialog()
        d.setWindowFlags(Qt.Popup)
        l = QHBoxLayout()
        d.setLayout(l)
        l.setContentsMargins(3,3,3,3)
        l.setSpacing(0)
        valueLineEdit = QLineEdit(d)
        l.addWidget(valueLineEdit)
        okButton = QToolButton(d)
        okButton.setIcon(CUI().icon('check'))
        okButton.clicked.connect(lambda b: d.accept())
        cancelButton = QToolButton(d)
        cancelButton.setIcon(CUI().icon('remove'))
        cancelButton.clicked.connect(lambda b: d.reject())
        l.addWidget(okButton)
        l.addWidget(cancelButton)

        okShortcut = QShortcut(QKeySequence(Qt.Key_Return), okButton)
        okShortcut.activated.connect(lambda: okButton.click())
        okShortcut2 = QShortcut(QKeySequence(Qt.Key_Enter), okButton)
        okShortcut2.activated.connect(lambda: okButton.click())
        cancelShortcut = QShortcut(QKeySequence(Qt.Key_Escape), cancelButton)
        cancelShortcut.activated.connect(lambda: cancelButton.click())

        valueLineEdit.setFocus()
        d.move(self.setIntValueButton.mapToGlobal(QPoint(0, self.setIntValueButton.height)))

        if d.exec_() != QDialog.Accepted:
            return

        sels = [sr.data(Qt.UserRole) for sr in self.intTable.selectionModel().selectedRows()]
        if not sels:
            sels = self.intModel.selections

        for sel in sels:
            if not sel.property('Internal element') or sel.property('Internal element') != 'Criteria':
                sel.setProperty('Internal value', float(valueLineEdit.text))
            else:
                sel.setProperty('Internal value', valueLineEdit.text)

        self.intModel.updateData()

    def processUnits(self, action):
        u = action.text
        sels = [sr.data(Qt.UserRole) for sr in self.intTable.selectionModel().selectedRows()]
        if not sels:
            sels = self.intModel.selections

        for sel in sels:
            sel.setProperty('Internal units', u)

        self.intModel.updateData()

    def editCriteria(self):
        d = CriteriaDialog()
        d.table.model().setCriteria(drs.setting('ISCriteria'))

        if d.exec_() == QDialog.Rejected:
            return

        drs.setSetting('ISCriteria', d.criteria())

    def updateDRSSetting(self, name, value):
        print(f'Updating DRS setting {name} to {value}')
        if drs.setting(name) == value:
            return
        drs.setSetting(name, value)
        self.processExtSelection()

    def toggleThroughZero(self, b):
        channels = [sr.data(Qt.UserRole) for sr in self.extTable.selectionModel().selectedRows()]
        if not channels:
            channels = [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c]

        for channel in channels:
            channel.setProperty('FitThroughZero', b)

        self.updateAffected()

        self.processExtSelection()
        self.extFilterModel.invalidate()

    def toggleFractionation(self, b):
        on = (isinstance(b, bool) and b) or (isinstance(b, QAction) and b.text != 'None')
        fit = 'Linear' if on else 'None'
        if on and isinstance(b, QAction):
            fit = b.text

        channels = [sr.data(Qt.UserRole) for sr in self.extTable.selectionModel().selectedRows()]
        if not channels:
            channels = [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]

        for channel in channels:
            channel.setProperty('FractionationCorrection', on)
            channel.setProperty('FractionationFitType', fit)

        self.processExtSelection()
        self.extFilterModel.invalidate()

    def changeModel(self, act):
        channels = [sr.data(Qt.UserRole) for sr in self.extTable.selectionModel().selectedRows()]
        if not channels:
            channels = [c for c in data.timeSeriesList(data.Input) if 'TotalBeam' not in c.name]

        for channel in channels:
            channel.setProperty('Model', self.modelComboBox.currentText)

        self.processExtSelection()
        self.extFilterModel.invalidate()

    def importValues(self):
        lastRawDataPath = QSettings().value('paths/lastrawdatapath', QDir().home().absolutePath())
        fileName = QFileDialog.getOpenFileName(self, 'Open IS file', lastRawDataPath, 'IS file (*.txt *.csv)')

        if not fileName:
            return

        res = data.importISValues(fileName)
        print(res)

        elements = res['elements']
        selsChanged = res['selsChanged']
        missedNames = res['missedNames']

        if selsChanged > 0 and len(elements) > 0:
            QMessageBox.information(self, "Internal standards", f'Internal standard data were imported for {selsChanged} selections and {len(elements)} elements.\n\nCould not match IS values for {len(missedNames)} selections', QMessageBox.Ok)
            self.intModel.updateData(self.intModel.selections)



def settingsWidget():
    drs.setSettingsWidget(SettingsWidget())


class MCLineEdit(QLineEdit):

    def __init__(self, completer, delimiter=',', parent=None):
        super().__init__(parent)
        self.mc = completer
        self.mc.setWidget(self)
        self.mc.setCaseSensitivity(Qt.CaseInsensitive)
        self.mc.setCompletionMode(QCompleter.UnfilteredPopupCompletion)
        self.mc.activated.connect(self.insertCompletion)
        self.delimiter = delimiter

    def keyPressEvent(self, event):
        QLineEdit.keyPressEvent(self, event)
        if not self.mc or self.text == '':
            return

        self.mc.setCompletionPrefix(self.cursorWord(self.text))
        if len(self.mc.completionPrefix) < 1:
            self.mc.popup().hide()
            return

        self.mc.complete()

    def cursorWord(self, sentence):
        p = sentence.rfind(self.delimiter)
        if p == -1:
            return sentence

        return sentence[p + 1:]

    def insertCompletion(self, text):
        p = self.text.rfind(self.delimiter)
        if p == -1:
            self.setText(text)
        else:
            self.setText(self.text[:p+1]+text)

        self.textEdited.emit(self.text)


class PTSettingsWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        settings = QSettings()
        self.ui_path = settings.value("Paths/DataReductionSchemesPath")
        self.ui_file = QFile(self.ui_path + "/3d_trace_elements_pt.ui")

        self.setLayout(QVBoxLayout())
        if not self.ui_file.open(QIODevice.ReadOnly):
            raise RuntimeError('Could not load settings ui')

        ui = QUiLoader().load(self.ui_file, self)
        self.layout().addWidget(ui)
        self.layout().setContentsMargins(0, 0, 0, 0)

        # Set DRS defaults:
        drs.setDefaultSetting("Mask", False)
        drs.setDefaultSetting('IndexChannel', '')
        drs.setDefaultSetting('MaskMethod', 'Laser log')
        drs.setDefaultSetting('MaskChannel', '')
        drs.setDefaultSetting("MaskCutoff", 0.1)
        drs.setDefaultSetting("MaskTrim", 0.0)
        drs.setDefaultSetting("NormalizeExternals", True)
        drs.setDefaultSetting('StatName', 'mean')
        drs.setDefaultSetting('UseFG', False)
        drs.setDefaultSetting("UseIntStds", True)
        drs.setDefaultSetting('SplineType', 'Spline_AutoSmooth')
        drs.setDefaultSetting('MasterExternal', '')
        drs.setDefaultSetting('BeamSecondsMethod', 'Laser log')
        drs.setDefaultSetting('BeamSecondsChannel', '')
        drs.setDefaultSetting('BeamSecondsValue', 1000.)
        drs.setDefaultSetting('AffinityCorrection', False)
        drs.setDefaultSetting('AffinityCorrection%', 15.0)
        drs.setDefaultSetting('BlockFindingMethod', 'Simple')
        drs.setDefaultSetting('NClusters', -1)
        drs.setDefaultSetting('PreserveISProperties', False)
        drs.setDefaultSetting('ISCriteria', [])

        # Get refs to UI elements:
        self.extTable = ui.findChild(QTableWidget, 'extTable')
        self.addExtButton = ui.findChild(QToolButton, 'addExtButton')
        self.removeExtButton = ui.findChild(QToolButton, 'removeExtButton')
        self.intGroupBox = ui.findChild(QGroupBox, 'intGroupBox')
        self.intTable = ui.findChild(QTableWidget, 'intTable')
        self.addIntButton = ui.findChild(QToolButton, 'addIntButton')
        self.removeIntButton = ui.findChild(QToolButton, 'removeIntButton')
        self.indexLineEdit = ui.findChild(QLineEdit, 'indexChannelLineEdit')
        self.maskCheckBox = ui.findChild(QCheckBox, 'maskCheckBox')
        self.maskMethodComboBox = ui.findChild(QComboBox, 'maskMethodComboBox')
        self.maskChannelLineEdit = ui.findChild(QLineEdit, "maskChannelLineEdit")
        self.maskChannelLabel = ui.findChild(QLabel, 'maskChannelLabel')
        self.maskTrimLineEdit = ui.findChild(QLineEdit, 'maskTrimLineEdit')
        self.maskTrimLabel = ui.findChild(QLabel, 'maskTrimLabel')
        self.maskValueLineEdit = ui.findChild(QLineEdit, 'maskValueLineEdit')
        self.maskValueLabel = ui.findChild(QLabel, 'maskValueLabel')
        self.blockComboBox = ui.findChild(QComboBox, 'blockComboBox')
        self.blockCountLabel = ui.findChild(QLabel, 'blockCountLabel')
        self.blockCountSpinBox = ui.findChild(QSpinBox, 'blockCountSpinBox')
        self.normalizeCheckBox = ui.findChild(QCheckBox, 'normalizeCheckBox')
        self.normalizeComboBox = ui.findChild(QComboBox, 'normalizeComboBox')
        self.splineComboBox = ui.findChild(QComboBox, 'splineComboBox')
        self.splineComboBox.addItems([
            'MeanMean',
            'MeanMedian',
            'LinearFit',
            'WeightedLinearFit',
            'StepLinear',
            'StepForward',
            'StepBackward',
            'StepAverage',
            'Nearest',
            'Akima',
            'Spline_NoSmoothing',
            'Spline_Smooth1',
            'Spline_Smooth2',
            'Spline_Smooth3',
            'Spline_Smooth4',
            'Spline_Smooth5',
            'Spline_Smooth6',
            'Spline_Smooth7',
            'Spline_Smooth8',
            'Spline_Smooth9',
            'Spline_Smooth10',
            'Spline_AutoSmooth'
        ])
        self.bsMethodComboBox = ui.findChild(QComboBox, 'bsMethodComboBox')
        self.bsChannelLineEdit = ui.findChild(QLineEdit, 'bsChannelLineEdit')
        self.bsChannelLabel = ui.findChild(QLabel, 'bsChannelLabel')
        self.bsValueLineEdit = ui.findChild(QLineEdit, 'bsValueLineEdit')
        self.bsValueLabel = ui.findChild(QLabel, 'bsValueLabel')
        self.affinityCheckBox = ui.findChild(QCheckBox, 'affinityCheckBox')
        self.affinitySpinBox = ui.findChild(QDoubleSpinBox, 'affinitySpinBox')
        self.criteriaButton = ui.findChild(QToolButton, 'criteriaButton')
        self.preserveISCheckBox = ui.findChild(QCheckBox, 'preserveISCheckBox')
        self.fgButton = ui.findChild(QToolButton, 'fgButton')

        # Setup UI elements:
        self.addExtButton.setIcon(CUI().icon('plus'))
        self.addIntButton.setIcon(CUI().icon('plus'))
        self.removeIntButton.setIcon(CUI().icon('minus'))
        self.removeExtButton.setIcon(CUI().icon('minus'))
        self.criteriaButton.setIcon(CUI().icon('edit'))
        tbs = [self.addExtButton, self.addIntButton, self.removeExtButton, self.removeIntButton, self.criteriaButton]
        [b.setToolButtonStyle(Qt.ToolButtonTextBesideIcon) for b in tbs]

        self.normalizeComboBox.addItems(data.referenceMaterialNames())
        self.blockComboBox.addItems(['Assigned', 'Simple', 'Clustering', 'Auto clustering'])

        self.extTable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.extTable.verticalHeader().setVisible(False)
        self.extTable.setSelectionBehavior(QAbstractItemView.SelectRows)

        self.intTable.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.intTable.verticalHeader().setVisible(False)
        self.intTable.setSelectionBehavior(QAbstractItemView.SelectRows)

        # Todo: Restore settings
        def showBlockCount(b):
            self.blockCountLabel.setVisible(b)
            self.blockCountSpinBox.setVisible(b)

        def showBSOptions(b):
            self.bsValueLabel.setVisible(b)
            self.bsValueLineEdit.setVisible(b)
            self.bsChannelLabel.setVisible(b)
            self.bsChannelLineEdit.setVisible(b)

        def showMaskOptions(b):
            self.maskChannelLabel.setVisible(b)
            self.maskChannelLineEdit.setVisible(b)
            self.maskValueLabel.setVisible(b)
            self.maskValueLineEdit.setVisible(b)
            self.maskTrimLabel.setVisible(b)
            self.maskTrimLineEdit.setVisible(b)

        def toggleMask(b):
            self.maskMethodComboBox.setEnabled(b)
            showMaskOptions(self.maskMethodComboBox.currentText=='Cutoff')

        self.splineComboBox.setCurrentText(drs.setting('SplineType'))
        self.normalizeCheckBox.setChecked(drs.setting('NormalizeExternals'))
        self.normalizeComboBox.setCurrentText(drs.setting('MasterExternal'))
        self.blockComboBox.setCurrentText(drs.setting('BlockFindingMethod'))
        self.blockCountSpinBox.setValue(drs.setting('NClusters'))
        self.intGroupBox.setChecked(drs.setting('UseIntStds'))
        self.indexLineEdit.setText(drs.setting('IndexChannel'))
        self.bsMethodComboBox.setCurrentText(drs.setting('BeamSecondsMethod'))
        self.bsChannelLineEdit.setText(drs.setting('BeamSecondsChannel'))
        self.bsValueLineEdit.setText(str(drs.setting('BeamSecondsValue')))
        self.maskCheckBox.setChecked(drs.setting('Mask'))
        self.maskMethodComboBox.setCurrentText(drs.setting('MaskMethod'))
        self.maskChannelLineEdit.setText(drs.setting('MaskChannel'))
        self.maskValueLineEdit.setText(str(drs.setting('MaskCutoff')))
        self.maskTrimLineEdit.setText(str(drs.setting('MaskTrim')))
        self.preserveISCheckBox.setChecked(drs.setting('PreserveISProperties'))

        showBlockCount(drs.setting('BlockFindingMethod') == 'Clustering')
        showBSOptions('threshold' in drs.setting('BeamSecondsMethod'))
        showMaskOptions(drs.setting('MaskMethod') == 'Cutoff')
        toggleMask(drs.setting('Mask'))

        if 'PTExternal' in drs.settings():
            for pt_ext in drs.setting('PTExternal'):
                self.addExt(pt_ext)
        else:
            self.addExt()

        if 'PTInternal' in drs.settings():
            for pt_is in drs.setting('PTInternal'):
                self.addInt(pt_is)
        else:
            self.addInt()

        # Make connections
        self.blockComboBox.textActivated.connect(lambda t: drs.setSetting('BlockFindingMethod', t))
        self.blockCountSpinBox.valueChanged.connect(lambda v: drs.setSetting('NClusters', int(v)))
        self.normalizeCheckBox.toggled.connect(lambda b: drs.setSetting('NormalizeExternals', b))
        self.normalizeComboBox.textActivated.connect(lambda t: drs.setSetting('MasterExternal', t))
        self.splineComboBox.textActivated.connect(lambda t: drs.setSetting('SplineType', t))
        self.bsMethodComboBox.textActivated.connect(lambda t: drs.setSetting('BeamSecondsMethod', t))
        self.affinityCheckBox.toggled.connect(lambda b: drs.setSetting('AffinityCorrection', b))
        self.affinitySpinBox.valueChanged.connect(lambda v: drs.setSetting('AffinityCorrection%', v))
        self.indexLineEdit.textEdited.connect(lambda t: drs.setSetting('IndexChannel', t))
        self.intGroupBox.toggled.connect(lambda b: drs.setSetting('UseIntStds', b))
        self.extTable.itemChanged.connect(self.updateExtInfo)
        self.intTable.itemChanged.connect(self.updateIntInfo)
        self.addExtButton.clicked.connect(lambda: self.addExt())
        self.removeExtButton.clicked.connect(self.removeExt)
        self.addIntButton.clicked.connect(lambda: self.addInt())
        self.removeIntButton.clicked.connect(self.removeInt)
        self.criteriaButton.clicked.connect(self.editCriteria)
        self.blockComboBox.textActivated.connect(lambda t: showBlockCount(t == 'Clustering'))
        self.bsMethodComboBox.textActivated.connect(lambda t: showBSOptions('threshold' in t))
        self.maskMethodComboBox.textActivated.connect(lambda t: showMaskOptions(t == 'Cutoff'))
        self.maskCheckBox.toggled.connect(lambda b: toggleMask(b))
        self.preserveISCheckBox.toggled.connect(lambda b: drs.setSetting('PreserveISProperties', b))

    def addExt(self, props=None):
        row = self.extTable.rowCount
        self.extTable.insertRow(row)

        nameItem = QTableWidgetItem()
        nameItem.setToolTip("Name of input channel")
        standardsItem = QTableWidgetItem()
        modelItem = QTableWidgetItem('ODR')
        zeroItem = QTableWidgetItem()
        zeroItem.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
        zeroItem.setCheckState(Qt.Unchecked)
        fracItem = QTableWidgetItem('None')

        if props is not None:
            nameItem.setText(props['name'])
            standardsItem.setText(props['standards'])
            modelItem.setText(props['model'])
            cs = Qt.Checked if props['zero'] else Qt.Unchecked
            zeroItem.setCheckState(cs)
            fracItem.setText(props['frac'])

        if row == 0:
            nameItem.setFlags(nameItem.flags() ^ Qt.ItemIsEditable)
            nameItem.setText('Default')

        self.extTable.setItem(row, 0, nameItem)
        self.extTable.setItem(row, 1, standardsItem)
        self.extTable.setItem(row, 2, modelItem)
        self.extTable.setItem(row, 3, zeroItem)
        self.extTable.setItem(row, 4, fracItem)

        completer = QCompleter(data.referenceMaterialNames())
        standardsLineEdit = MCLineEdit(completer, ',')
        standardsLineEdit.setText(standardsItem.text())
        standardsLineEdit.setToolTip("List RMs here (comma separated)")
        self.extTable.setCellWidget(row, 1, standardsLineEdit)
        standardsLineEdit.textEdited.connect(lambda t: standardsItem.setText(t))

        modelComboBox = QComboBox()
        modelComboBox.addItems(['ODR', 'OLS', 'WLS', 'RLM', 'York'])
        modelComboBox.setCurrentText(modelItem.text())
        self.extTable.setCellWidget(row, 2, modelComboBox)
        modelComboBox.textActivated.connect(lambda t: modelItem.setText(t))

        fracComboBox = QComboBox()
        fracComboBox.addItems(['None', 'Linear', 'Spline'])
        fracComboBox.setCurrentText(fracItem.text())
        self.extTable.setCellWidget(row, 4, fracComboBox)
        fracComboBox.textActivated.connect(lambda t: fracItem.setText(t))

    def removeExt(self):
        rows = self.extTable.selectionModel().selectedRows()
        if len(rows) > 0:
            ri = rows[0].row()
            if ri == 0:
                QMessageBox.information(self, 'Remove external', 'Cannot remove the default row.')
                return
            self.extTable.removeRow(ri)

    def updateExtInfo(self):
        info = []

        for row in range(self.extTable.rowCount):
            info.append({
                'name': self.extTable.item(row, 0).text(),
                'standards': self.extTable.item(row, 1).text(),
                'model': self.extTable.item(row, 2).text(),
                'zero': self.extTable.item(row, 3).checkState() == Qt.Checked,
                'frac': self.extTable.item(row, 4).text()
            })
        drs.setSetting('PTExternal', info)

    def addInt(self, props=None):
        row = self.intTable.rowCount
        self.intTable.insertRow(row)

        groupItem = QTableWidgetItem()
        selectionItem = QTableWidgetItem()
        elementsItem = QTableWidgetItem()
        valueItem = QTableWidgetItem()
        unitsItem = QTableWidgetItem()
        affinityItem = QTableWidgetItem()

        if props is not None:
            groupItem.setText(props['group'])
            selectionItem.setText(props['selection'])
            elementsItem.setText(props['elements'])
            valueItem.setText(props['value'])
            unitsItem.setText(props['units'])
            affinityItem.setText(props['affinity'])

        if row == 0:
            groupItem.setFlags(groupItem.flags() ^ Qt.ItemIsEditable)
            groupItem.setText('Default')
            selectionItem.setFlags(selectionItem.flags() ^ Qt.ItemIsEditable)
            selectionItem.setText('Default')

        self.intTable.setItem(row, 0, groupItem)
        self.intTable.setItem(row, 1, selectionItem)
        self.intTable.setItem(row, 2, elementsItem)
        self.intTable.setItem(row, 3, valueItem)
        self.intTable.setItem(row, 4, unitsItem)
        self.intTable.setItem(row, 5, affinityItem)

        unitsComboBox = QComboBox()
        unitsComboBox.addItems(['ppm', 'ppb', 'wtpc', 'wtpc_oxide'])
        unitsComboBox.setCurrentText(unitsItem.text())
        self.intTable.setCellWidget(row, 4, unitsComboBox)
        unitsComboBox.textActivated.connect(lambda t: unitsItem.setText(t))

    def removeInt(self):
        rows = self.intTable.selectionModel().selectedRows()
        if len(rows) > 0:
            ri = rows[0].row()
            if ri == 0:
                QMessageBox.information(self, 'Remove internal', 'Cannot remove the default row.')
                return
            self.intTable.removeRow(ri)

    def updateIntInfo(self):
        info = []
        for row in range(self.intTable.rowCount):
            info.append({
                'group': self.intTable.item(row, 0).text(),
                'selection': self.intTable.item(row, 1).text(),
                'elements': self.intTable.item(row, 2).text(),
                'value': self.intTable.item(row, 3).text(),
                'units': self.intTable.item(row, 4).text(),
                'affinity': self.intTable.item(row, 5).text()
            })
        drs.setSetting('PTInternal', info)

    def editCriteria(self):
        d = CriteriaDialog()
        d.table.model().setCriteria(drs.setting('ISCriteria'))

        if d.exec_() == QDialog.Rejected:
            return

        drs.setSetting('ISCriteria', d.criteria())

def ptSettingsWidget():
    sw = PTSettingsWidget()
    drs.setPTSettingsWidget(sw)
