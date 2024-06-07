import eazy

import os
import glob
import yaml
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import grizli
from grizli import utils, jwst_utils
jwst_utils.set_quiet_logging()
utils.set_warnings()

import astropy.io.fits as pyfits
from jwst.datamodels import SlitModel


import msaexp
from msaexp import pipeline
import msaexp.drizzle
import msaexp.spectrum
import astropy.units as u

msaexp.spectrum.FFTSMOOTH = True
#reload(msaexp.drizzle)
#reload(msaexp.utils)

print(f'grizli version = {grizli.__version__}')
print(f'msaexp version = {msaexp.__version__}')
print(f'matplotlib version = {matplotlib.__version__}')

def make_templates(sampler, z, bspl={}, eazy_templates=None, vel_width=100, broad_width=4000, 
                   broad_lines=[], scale_disp=1.3, use_full_dispersion=False, disp=None, grating='prism', 
                   halpha_prism=['Ha+NII'], oiii=['OIII'], o4363=[], sii=['SII'], lorentz=False, with_pah=True,unlock_ratios=False,exclude_lines=[], **kwargs):
    """
    Generate fitting templates
    
    wobs : array
        Observed-frame wavelengths of the spectrum to fit, microns
    
    z : float
        Redshift
    
    bspl : dict
        Spline templates for dummy continuum
    
    eazy_templates : list
        Optional list of `eazy.templates.Template` template objects to use in 
        place of the spline + line templates
    
    vel_width : float
        Velocity width of the individual emission line templates
    
    halpha_prism : ['Ha+NII'], ['Ha','NII']
        Line template names to use for Halpha and [NII], i.e., ``['Ha+NII']`` 
        fits with a fixed line ratio and `['Ha','NII']` fits them separately 
        but with a fixed line ratio 6548:6584 = 1:3
    
    oiii : ['OIII'], ['OIII-4959','OIII-5007']
        Similar for [OIII]4959+5007, ``['OIII']`` fits as a doublet with fixed
        ratio 4959:5007 = 1:2.98 and ``['OIII-4949', 'OIII-5007']`` fits them
        independently.
    
    o4363 : [] or ['OIII-4363']
        How to fit [OIII]4363.
    
    sii : ['SII'], ['SII-6717','SII-6731']
        [SII] doublet
    
    lorentz : bool
        Use Lorentzian profile for lines
    
    Returns
    -------
    templates : list
        List of the computed template objects
    
    tline : array
        Boolean list of which templates are line components
    
    _A : (NT, NWAVE) array
        Design matrix of templates interpolated at `wobs`
    
    """
    from grizli import utils
    import eazy.igm
    igm = eazy.igm.Inoue14()

    
    wobs = sampler.spec_wobs
    wrest = wobs/(1+z)*1.e4
    
    spec = sampler.spec
    wmask = sampler.valid
    # wmask = np.isfinite(spec['flux'])
    # wmask = (spec['err'] > 0) & (spec['flux'] != 0)
    # wmask &= np.isfinite(spec['err']+spec['flux'])
    
    wmin = wobs[wmask].min()
    wmax = wobs[wmask].max()
    
    templates = []
    tline = []
    
    if eazy_templates is None:
        lw, lr = utils.get_line_wavelengths()
        
        _A = [bspl*1]
        for i in range(bspl.shape[0]):
            templates.append(f'spl {i}')
            tline.append(False)
            
        #templates = {}
        #for k in bspl:
        #    templates[k] = bspl[k]

        # templates = {}
        if grating in ['prism']:
            hlines = ['Hb', 'Hg', 'Hd']
            
            if z > 4:
                oiii = ['OIII-4959','OIII-5007']
                hene = ['HeII-4687', 'NeIII-3867','HeI-3889']
                o4363 = ['OIII-4363']
                
            else:
                #oiii = ['OIII']
                hene = ['HeI-3889']
                #o4363 = []
                
            #sii = ['SII']
            #sii = ['SII-6717', 'SII-6731']
            
            hlines += halpha_prism + ['NeIII-3968']
            fuv = ['OIII-1663']
            oii_7320 = ['OII-7325']
            extra = []
            
        else:
            hlines = ['Hb', 'Hg', 'Hd','H8','H9', 'H10', 'H11', 'H12']
            
            hene = ['HeII-4687', 'NeIII-3867']
            o4363 = ['OIII-4363']
            oiii = ['OIII-4959','OIII-5007']
            sii = ['SII-6717', 'SII-6731']
            hlines += ['Ha', 'NII-6549', 'NII-6584']
            hlines += ['H7', 'NeIII-3968']
            fuv = ['OIII-1663', 'HeII-1640', 'CIV-1549']
            oii_7320 = ['OII-7323', 'OII-7332']
            
            extra = ['HeI-6680', 'SIII-6314','NeIV']
            
        line_names = []
        line_waves = []
        
        for l in [*hlines, *oiii, *o4363, 'OII',
                  *hene, 
                  *sii,
                  *oii_7320,
                  'ArIII-7138', 'ArIII-7753', 'SIII-9068', 'SIII-9531',
                  'OI-6302', 'PaD', 'PaG', 'PaB', 'PaA', 'HeI-1083',
                  'BrA','BrB','BrG','BrD','PfB','PfG','PfD','PfE',
                  'Pa8','Pa9','Pa10',
                  'HeI-5877', 
                  *fuv,
                  'CIII-1906', 'NIII-1750', 'Lya',
                  'MgII', 'NeV-3346', 'NeVI-3426',
                  'HeI-7065', 'HeI-8446',
                  *extra
                   ]:

            if l in exclude_lines:
                continue

            if l not in lw:
                continue
            
            lwi = lw[l][0]*(1+z)

            if lwi < wmin*1.e4:
                continue

            if lwi > wmax*1.e4:
                continue
            
            line_names.append(l)
            line_waves.append(lwi)
        
        so = np.argsort(line_waves)
        line_waves = np.array(line_waves)[so]

        if unlock_ratios:
            for iline in so:
                l = line_names[iline]
                lwi = lw[l][0]*(1+z)
                if lwi < wmin*1.e4:
                    continue

                if lwi > wmax*1.e4:
                    continue

                lwi0 = lw[l][0]
                name = f'line {l}'
                lwi = lwi0*(1+z)/1.e4
                if l in broad_lines:
                    vel_i = broad_width
                else:
                    vel_i = vel_width

                line_i = sampler.fast_emission_line(lwi,
                                    line_flux=1,
                                    scale_disp=scale_disp,
                                    velocity_sigma=vel_i,)
                line_0 = line_i

                _A.append(line_0/1.e4)
                templates.append(name)
                tline.append(True)

        if not unlock_ratios:
            for iline in so:
                l = line_names[iline]
                lwi = lw[l][0]*(1+z)

                if lwi < wmin*1.e4:
                    continue

                if lwi > wmax*1.e4:
                    continue
                
                # print(l, lwi, disp_r)

                name = f'line {l}'

                for i, (lwi0, lri) in enumerate(zip(lw[l], lr[l])):
                    lwi = lwi0*(1+z)/1.e4
                    if l in broad_lines:
                        vel_i = broad_width
                    else:
                        vel_i = vel_width
                        
                    line_i = sampler.fast_emission_line(lwi,
                                        line_flux=lri/np.sum(lr[l]),
                                        scale_disp=scale_disp,
                                        velocity_sigma=vel_i,)
                    if i == 0:
                        line_0 = line_i
                    else:
                        line_0 += line_i
                    
                _A.append(line_0/1.e4)
                templates.append(name)
                tline.append(True)
        
        if with_pah:
            xpah = 3.3*(1+z)
            if ((xpah > wmin) & (xpah < wmax)) | (0):
                for t in PAH_TEMPLATES:
                    tp = PAH_TEMPLATES[t]
                    tflam = sampler.resample_eazy_template(tp,
                                            z=z,
                                            velocity_sigma=vel_width,
                                            scale_disp=scale_disp,
                                            fnu=False)
            
                    _A.append(tflam)
            
                    templates.append(t)
                    tline.append(True)
                    
                
        _A = np.vstack(_A)
        
        ll = wobs.value*1.e4/(1+z) < 1215.6

        igmz = igm.full_IGM(z, wobs.value*1.e4)
        _A *= np.maximum(igmz, 0.01)
        
    else:
        if isinstance(eazy_templates[0], dict) & (len(eazy_templates) == 2):
            # lw, lr dicts
            lw, lr = eazy_templates
            
            _A = [bspl*1]
            for i in range(bspl.shape[0]):
                templates.append(f'spl {i}')
                tline.append(False)
            
            for l in lw:
                name = f'line {l}'
                
                line_0 = None
                
                for i, (lwi0, lri) in enumerate(zip(lw[l], lr[l])):
                    lwi = lwi0*(1+z)/1.e4
                    
                    if lwi < wmin:
                        continue

                    elif lwi > wmax:
                        continue
                    
                    if l in broad_lines:
                        vel_i = broad_width
                    else:
                        vel_i = vel_width
                    
                    line_i = sampler.fast_emission_line(lwi,
                                        line_flux=lri/np.sum(lr[l]),
                                        scale_disp=scale_disp,
                                        velocity_sigma=vel_i,)
                    if line_0 is None:
                        line_0 = line_i
                    else:
                        line_0 += line_i
                
                if line_0 is not None:
                    _A.append(line_0/1.e4)
                    templates.append(name)
                    tline.append(True)
            
            _A = np.vstack(_A)
        
            ll = wobs.value*1.e4/(1+z) < 1215.6

            igmz = igm.full_IGM(z, wobs.value*1.e4)
            _A *= np.maximum(igmz, 0.01)
        
        elif len(eazy_templates) == 1:
            # Scale single template by spline
            t = eazy_templates[0]
            
            for i in range(bspl.shape[0]):
                templates.append(f'{t.name} spl {i}')
                tline.append(False)
            
            tflam = sampler.resample_eazy_template(t,
                                    z=z,
                                    velocity_sigma=vel_width,
                                    scale_disp=scale_disp,
                                    fnu=False)
            
            _A = np.vstack([bspl*tflam])
        
            ll = wobs.value*1.e4/(1+z) < 1215.6

            igmz = igm.full_IGM(z, wobs.value*1.e4)
            _A *= np.maximum(igmz, 0.01)
            
        else:
            templates = []
            tline = []
        
            _A = []
            for i, t in enumerate(eazy_templates):
                tflam = sampler.resample_eazy_template(t,
                                        z=z,
                                        velocity_sigma=vel_width,
                                        scale_disp=scale_disp,
                                        fnu=False)
            
                _A.append(tflam)
            
                templates.append(t.name)
                tline.append(False)
            
            _A = np.vstack(_A)
            
    return templates, np.array(tline), _A

def make_broad_templates(sampler, z, broad_width=4000, broad_lines=[], scale_disp=1.3, use_full_dispersion=False, disp=None, grating='prism'):

    import eazy.igm
    igm = eazy.igm.Inoue14()

    from grizli import utils
    
    wobs = sampler.spec_wobs
    wrest = wobs/(1+z)*1.e4
    
    spec = sampler.spec
    wmask = sampler.valid
    # wmask = np.isfinite(spec['flux'])
    # wmask = (spec['err'] > 0) & (spec['flux'] != 0)
    # wmask &= np.isfinite(spec['err']+spec['flux'])
    
    wmin = wobs[wmask].min()
    wmax = wobs[wmask].max()
    
    templates = []
    tline = []
    _A = []
    
    lw, lr = utils.get_line_wavelengths()

    for l in broad_lines:
        name = f'line {l} broad'
        if l not in lw:
            print(l,'not in the line list, make sure the name is right')
            continue

        lwi0 = lw[l][0]
        lwi = lwi0*(1+z)/1.e4

        if lwi < wmin:
            continue

        if lwi > wmax:
            continue
        
        line_i = sampler.fast_emission_line(lwi,
                                            line_flux=1.,
                                            scale_disp=scale_disp,
                                            velocity_sigma=broad_width,)
        
        line_0 = line_i
        
                
        _A.append(line_0/1.e4)
        templates.append(name)
        tline.append(True)

    if len(_A)==0:
        _A.append(np.zeros_like(wobs))
    _A = np.vstack(_A)
        
    ll = wobs.value*1.e4/(1+z) < 1215.6

    igmz = igm.full_IGM(z, wobs.value*1.e4)
    _A *= np.maximum(igmz, 0.01)

    return templates, np.array(tline), _A


def adjust_spectrum(sampler):
    # sampler.valid = np.ones_like(sampler.valid,dtype=bool)
    # spec = sampler.spec
    for i,fl in enumerate(sampler.spec['flux']):
        if i==0:
            continue
        if sampler.spec['flux'][i]==0:
            sampler.spec['flux'][i] = sampler.spec['flux'][i-1]
    return sampler

def fit_spectrum(file,nspline,zgrid,narrow_grid,broad_grid,broad_lines,scale_disp=1.3,id=None,
                 custom_range=None,unlock_ratios=False,save_data=False,err_thresh=1.2,method='lstsq',correct_spectrum=False,
                 exclude_lines=[],first_pass=False):
    import yaml
    from tqdm import tqdm
    import msaexp

    msaexp.spectrum.SCALE_UNCERTAINTY = [0.]
    

    sampler = msaexp.spectrum.SpectrumSampler(file,err_mask=None, err_median_filter=None)

    if correct_spectrum:
        sampler = adjust_spectrum(sampler)

    spec = sampler.spec



    print('Grating:',spec.grating)


    try:
        src_id = sampler.meta['SRCNAME']
        # sampler.meta['FILE1'].split('.')[-2].split('_')[1]
    except:
        if id is not None:
            src_id=id
        else:
            src_id = 0
    bspl = sampler.bspline_array(nspline=nspline, get_matrix=True)

    if first_pass:
        print('First pass of the models to get redshift only')
        print('Ignoring all input params, setting zgrid to 0.1<z<20')
        zgrid = np.arange(0.1,20,.2)
        narrow_grid = np.arange(249,250,1)
        broad_grid = np.arange(249,250,1)
    



    # coeffs_fit = np.zeros((len(zgrid),len(narrow_grid),len(broad_grid)))
    chi2_fit = np.zeros((len(zgrid),len(narrow_grid),len(broad_grid)))

    print('Fitting a total of', len(chi2_fit.flatten()), 'models')

    if custom_range is not None:
        print('Custom Fitting Range')
        mask_new = (spec['wave']>custom_range[0]) & (spec['wave']<custom_range[1])
        spec['valid'][~mask_new]=0
        sampler.valid[~mask_new]=False

    if unlock_ratios:
        print('Line ratios ignored')

    if len(exclude_lines)>0:
        print('The following lines will not be fit with a narrow component - ',exclude_lines)


    wobs = spec['wave']
    mask = sampler.valid


    flam = spec['flux']*spec['to_flam']
    eflam = spec['full_err']*spec['to_flam']

    flam[~mask] = np.nan
    eflam[~mask] = np.nan

    if len(broad_lines)==0:
        print('No Broad Lines Defined: Only fitting narrow lines')

    # Loop to create and fit all models

    for i, z_i in tqdm(enumerate(zgrid)):
        for j,fwhm_n_i in enumerate(narrow_grid):
            for k,fwhm_b_i in enumerate(broad_grid):

                # cntr+=1

                vn_i = fwhm_n_i/(2*np.sqrt(2*np.log(2)))
                vb_i = fwhm_b_i/(2*np.sqrt(2*np.log(2)))
                
                # templates_narrow, tline_narrow, _A_narrow = msaexp.spectrum.make_templates(sampler,z=z_i,bspl=bspl,vel_width=vn_i,scale_disp =scale_disp,grating=spec.grating)
                templates_narrow, tline_narrow, _A_narrow = make_templates(sampler,z=z_i,bspl=bspl,vel_width=vn_i,
                                                                           scale_disp=scale_disp,grating=spec.grating,unlock_ratios=unlock_ratios,exclude_lines=exclude_lines,)
                templates_broad, tline_broad, _A_broad = make_broad_templates(sampler,z=z_i, broad_width=vb_i,
                                                                               broad_lines=broad_lines,scale_disp =scale_disp,grating=spec.grating)
                
                # templates,tline, _A =  make_templates(sampler,z=z,narrow_fwhm=vn,broad_fwhm=vb,broad_lines=broad_lines,scale_disp=1.3,grating=spec.grating,bspl=bspl,oversamp=oversamp)
                # templates_broad,tline_broad, _A_broad =  make_templates(sampler,z=z,narrow_fwhm=vn,broad_fwhm=vb,broad_lines=broad_lines,just_broad=True,scale_disp=1.3,grating=spec.grating,bspl=bspl,oversamp=oversamp)

                if len(templates_broad)>0:
                    templates = templates_narrow+templates_broad
                    tline =  np.append(tline_narrow, tline_broad)
                    _A = np.concatenate((_A_narrow,_A_broad))
                else:
                    templates = templates_narrow
                    tline = tline_narrow
                    _A = _A_narrow
    
                okt = _A[:,mask].sum(axis=1) > 0
        
                _Ax = _A[okt,:]/eflam
                _yx = flam/eflam

                if method=='nnls':
                    _x = nnls(_Ax[:,mask].T, _yx[mask])
                elif method=='lstsq':
                    _x = np.linalg.lstsq(_Ax[:,mask].T,_yx[mask], rcond=None)

                else:
                    print('Choose a valid fitting method: 1) lstsq 2) nnls')
        
                coeffs = np.zeros(_A.shape[0])
                coeffs[okt] = _x[0]

                _model = _A.T.dot(coeffs)

                chi = (flam - _model) / eflam

                chi2_i = (chi[mask]**2).sum()

                chi2_fit[i,j,k] = chi2_i
                # coeffs_fit[i,j,k] = coeffs

    best_idx = np.unravel_index(np.argmin(chi2_fit),chi2_fit.shape)
    dof = len(_yx[mask]-len(templates))


    # Compute uncertainty on parameters
    # --------------------------------------------
    chi2_fit_reduced = chi2_fit/dof
    err_thresh = err_thresh
    _idx_distr = np.argwhere(np.abs(chi2_fit_reduced-chi2_fit[best_idx]/dof)<err_thresh)

    z_distr = []
    fwhm_narrow_distr = []
    fwhm_broad_distr = []


    for _idx in _idx_distr:
        z_distr.append(zgrid[_idx[0]])
        fwhm_narrow_distr.append(narrow_grid[_idx[1]])
        fwhm_broad_distr.append(broad_grid[_idx[2]])


    z_percentile = np.nanpercentile(z_distr,q=[16,50,84])
    fwhm_narrow_percentile = np.nanpercentile(fwhm_narrow_distr,q=[16,50,84])
    fwhm_broad_percentile = np.nanpercentile(fwhm_broad_distr,q=[16,50,84])

    # print(z_percentile)
    # print(fwhm_narrow_percentile)
    # print(fwhm_broad_percentile)
    
    sigma_z = np.median([z_percentile[2]-z_percentile[1],z_percentile[1]-z_percentile[0]])
    sigma_fwhm_narrow = np.median([fwhm_narrow_percentile[2]-fwhm_narrow_percentile[1],fwhm_narrow_percentile[1]-fwhm_narrow_percentile[0]])
    sigma_fwhm_broad = np.median([fwhm_broad_percentile[2]-fwhm_broad_percentile[1],fwhm_broad_percentile[1]-fwhm_broad_percentile[0]])
    # --------------------------------------------

                                

    zbest = zgrid[best_idx[0]]
    fwhm_n_best = narrow_grid[best_idx[1]]
    fwhm_b_best = broad_grid[best_idx[2]]
    # zbest = z_percentile[1]
    # fwhm_n_best = fwhm_narrow_percentile[1]
    # fwhm_b_best = fwhm_broad_percentile[1]


    vn_best = fwhm_n_best/(2*np.sqrt(2*np.log(2)))
    vb_best = fwhm_b_best/(2*np.sqrt(2*np.log(2)))


    # Obtained best-fit

    print('Best chi2',chi2_fit[best_idx])
    print('Best reduced chi2',chi2_fit[best_idx]/dof)
    print('Best z',zbest,'+-',sigma_z,)
    print('FWHM_narrow', fwhm_n_best,'+-',sigma_fwhm_narrow,'km/s')
    print('FWHM_broad', fwhm_b_best,'+-',sigma_fwhm_broad,'km/s')

    # Get best fit templ
                
    # templates_narrow, tline_narrow, _A_narrow = msaexp.spectrum.make_templates(sampler,z=zbest,bspl=bspl,vel_width=vn_best,scale_disp = scale_disp,grating=spec.grating)
    templates_narrow, tline_narrow, _A_narrow = make_templates(sampler,z=zbest,bspl=bspl,vel_width=vn_best,scale_disp = scale_disp,grating=spec.grating,unlock_ratios=unlock_ratios,
                                                               exclude_lines=exclude_lines)
    templates_broad, tline_broad, _A_broad = make_broad_templates(sampler,zbest, broad_width=vb_best, broad_lines=broad_lines,scale_disp = scale_disp,grating=spec.grating)

    
    if len(templates_broad)>0:
        templates = templates_narrow+templates_broad
        tline =  np.append(tline_narrow, tline_broad)
        _A = np.concatenate((_A_narrow,_A_broad))
    else:
        templates = templates_narrow
        tline = tline_narrow
        _A = _A_narrow


    mask_broad = np.zeros_like(tline,dtype=bool)
    if len(templates_broad)>0:
        mask_broad[-len(templates_broad):] = True

    
    okt = _A[:,mask].sum(axis=1) > 0

    _Ax = _A[okt,:]/eflam
    _yx = flam/eflam

    if method=='nnls':
        _x = nnls(_Ax[:,mask].T, _yx[mask])
    elif method=='lstsq':
        _x = np.linalg.lstsq(_Ax[:,mask].T,_yx[mask], rcond=None)

    coeffs = np.zeros(_A.shape[0])
    coeffs[okt] = _x[0]

    wmask = sampler.valid

    _model = _A.T.dot(coeffs)
    _mline = _A.T.dot(coeffs*tline)
    _mline_narrow = _A.T.dot(coeffs*tline*~mask_broad)
    _mline_broad = _A.T.dot(coeffs*tline*mask_broad)
    _mcont = _model - _mline

    full_chi2 = ((flam - _model)**2/eflam**2)[mask].sum()
    cont_chi2 = ((flam - _mcont)**2/eflam**2)[mask].sum()

    _model[~wmask] = np.nan
    _mline[~wmask] =  np.nan
    _mline_narrow[~wmask] = np.nan
    _mline_broad[~wmask] = np.nan
    _mcont[~wmask] = np.nan
    

    # input = {'lambda':wobs,'flam':flam,'eflam':eflam}
    # models = {'lambda':wobs,'full':_model,'cont':_mcont,'narrow':_mline_narrow,'broad':_mline_broad}
    # best_fit = {'id':src_id,'grating':spec.grating,'zbest':zbest,'fwhm_n':fwhm_n_best,'fwhm_b':fwhm_b_best}


    lw, lr = utils.get_line_wavelengths()

    try:
        oktemp = okt & (coeffs != 0)
            
        AxT = (_A[oktemp,:]/eflam)[:,mask].T
    
        covar_i = utils.safe_invert(np.dot(AxT.T, AxT))
        covar = utils.fill_masked_covar(covar_i, oktemp)
        covard = np.sqrt(covar.diagonal())
            
        has_covar = True
    except:
        has_covar = False
        covard = coeffs*0.
        N = len(templates)
        covar = np.eye(N, N)


    print(f'\n# line flux err\n# flux x 10^-20 erg/s/cm2')
    
    print(f'# z = {zbest:.5f}\n# {time.ctime()}')
    
    cdict = {}
    eqwidth = {}
    
    for i, t in enumerate(templates):
        cdict[t] = [float(coeffs[i]), float(covard[i])]
        if t.startswith('line '):
            lk = t.split()[-1]

            if 'broad' in t:
                 lk = t.split()[-2]

            
            # Equivalent width:
            # coeffs, line fluxes are in units of 1e-20 erg/s/cm2
            # _mcont, continuum model is in units of 1-e20 erg/s/cm2/A
            # so observed-frame equivalent width is roughly
            # eqwi = coeffs[i] / _mcont[ wave_obs[i] ]
            
            if lk in lw:
                lwi = lw[lk][0]*(1+zbest)/1.e4
                continuum_i = np.interp(lwi, spec['wave'], _mcont)
                eqwi = coeffs[i]/continuum_i
            else:
                eqwi = np.nan
            
            eqwidth[t] = [float(eqwi)]
            
            print(f'{t:>20}   {coeffs[i]:8.1f} ± {covard[i]:8.1f} (EW={eqwi:9.1f})')

    # 'ra': float(spec.meta['srcra']),
    # 'dec': float(spec.meta['srcdec']),
    # 'name': str(spec.meta['srcname']),
    data = {}
    data = {'z': float(zbest),
            'err_z': float(sigma_z),
            'id':src_id,
            'fwhm_narrow':float(fwhm_n_best),
            'err_fwhm_narrow':float(sigma_fwhm_narrow),
            'fwhm_broad':float(fwhm_b_best),
            'err_fwhm_broad':float(sigma_fwhm_broad),
            'file':file,
            'label':None,
            'grating':spec.grating,
            'wmin':float(spec['wave'][mask].min()),
            'wmax':float(spec['wave'][mask].max()),
            'coeffs':cdict,
            'covar':covar.tolist(),
            'wave': [float(m) for m in spec['wave']],
            'flux': [float(m) for m in spec['flux']],
            'err': [float(m) for m in spec['err']],
            'flam': [float(m) for m in spec['flux']*spec['to_flam']],
            'eflam': [float(m) for m in spec['err']*spec['to_flam']],
            'escale': [float(m) for m in spec['escale']],
            'model': [float(m) for m in _model],
            'mline':[float(m) for m in _mline],
            'mline_narrow':[float(m) for m in _mline_narrow],
            'mline_broad':[float(m) for m in _mline_broad],
            'mcont': [float(m) for m in _mcont],
            'templates':templates, 
            'dof': int(mask.sum()), 
            'fullchi2': float(full_chi2), 
            'contchi2': float(cont_chi2),
            'chi2red': float(chi2_fit[best_idx]/dof),
            'obs_eqwidth': eqwidth,
           }

    # for k in ['coeffs', 'covar', 'model', 'mline', 'fullchi2', 'contchi2','eqwidth']:
    #         if k in spl_data:
    #             data[f'spl_{k}'] = spl_data[k]
    
    if 'spec.fits' in file:
        froot = file.split('.spec.fits')[0]
    else:
        froot = file.split('.fits')[0]


    if save_data:
        if unlock_ratios:
            suffix = '.ratios_unlocked'
        else:
            suffix = ''
        if len(broad_lines)==0:
            suffix+='.narrow_only'
        with open(froot+suffix+'.msaexp_broad.yaml', 'w') as fp:
            yaml.dump(data, stream=fp)


    return data,sampler

            
def resample_custom_template(sampler,z,template,scale_disp,velocity_sigma,nsig=4,fnu=False):
    # Resamples the input tempalte to the spectral resolution given in the sampler

    templ_wobs = template['wave']*(1+z)/1e4
    templ_flux = template['flam']

    res = sampler.resample_func(sampler.spec_wobs,
                                 sampler.spec_R_fwhm*scale_disp,
                                 templ_wobs,
                                 templ_flux,
                                 velocity_sigma=velocity_sigma,
                                 nsig=nsig)
    
    return res
