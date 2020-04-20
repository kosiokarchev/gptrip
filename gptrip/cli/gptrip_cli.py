import os
from functools import partial
from time import time

import torch, numpy as np
from matplotlib import pyplot as plt, colors

from ..utils import compose
from ..torch import utils as torchutils
from .. import GPTrip
from ..torch import GPTripTorch
from ..song import Song

import click
from click_option_group import optgroup
from inspect import signature


def signature_to_decs(sig: signature, filter=lambda param: True, grouped=False,
                      defaults_in_help=True):
    arguments, options = dict(), dict()
    for param in sig.parameters.values():
        if filter(param) and param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL):
            if param.default is param.empty:
                arguments[param.name] = click.argument(param.name.upper())
            else:
                options[param.name] = (optgroup if grouped else click).option(
                    '--'+param.name, default=param.default, is_flag=type(param.default)==bool,
                    show_default=defaults_in_help
                )
    return options, arguments


def call(func, params, *args, **kwargs):
    params = params.copy()
    params.update(kwargs)
    sig_params = signature(func).parameters
    # print(args, {key: val for key, val in params.items() if key in sig_params})
    return func(*args, **{key: val for key, val in params.items() if key in sig_params})


RES = {
    'hd': (1280, 720),
    'fullhd': (1920, 1080),
    '4k': (3840, 2160)
}


def main(inname,
         outname='{inname}-{res}{fps}-{cmap}{ncolors}.mp4',
         res='hd', do_norm=False, fftshift=False,
         device=0, seed=42,
         coherent=False, exact=None, dynamic=0., spread=1.,
         cmap='hsv', ncolors=6, vmin=-3., vmax=3.,
         crf=11, **kwargs):
    t0 = time()
        
    print({'inname': inname, 'outname': outname,
           'res': res, 'do_norm': do_norm, 'crf': crf,
           'device': 0, 'seed': seed,
           'coherent': coherent, 'exact': exact, 'dynamic': dynamic, 'spread': spread,
           'cmap': cmap, 'ncolors': ncolors, 'vmin': vmin, 'vmax': vmax})
    print(kwargs)

    outname = outname.format(res=res, cmap=cmap, ncolors=ncolors, crf=crf,
                             norm='sum' if do_norm else 'nonorm',
                             fps=kwargs['specrate'],
                             inname=os.path.splitext(inname)[0],
                             basename=os.path.basename(os.path.splitext(inname)[0]))
    try:
        cmap = colors.LinearSegmentedColormap.from_list('cmap', eval(cmap), N=ncolors)
    except:
        cmap = plt.get_cmap(cmap, ncolors)

    use_torch = torch.cuda.is_available() and 0 <= device < torch.cuda.device_count()

    if use_torch:
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        GPClass = GPTripTorch
        norm = torchutils.Normalize
        cmap = torchutils.SampledColormap(cmap, ncolors)
    else:
        GPClass = GPTrip
        norm = plt.Normalize

    norm = norm(vmin=vmin, vmax=vmax)

    song = call(Song, kwargs, inname=inname)
    gpt = call(GPClass, kwargs, song, *RES[res])

    if coherent:
        gpt.coherent()
        fftshift = True

    if exact is not None:
        gpt.exact(exact)

    if dynamic > 0:
        gpt.interpolate_directions(dynamic, spread)

    if do_norm:
        song.normalize()

    t1 = time()

    gpt.render(
        fftshift=fftshift,
        cmap=cmap, norm=norm,
        vidname=outname, crf=crf
    )

    tf = time()
    size = os.path.getsize(outname) / 2**30
    print(f'Rendered {outname} ({size:.2f} GB) in {tf-t1:.0f}s ({tf-t0:.0f}s wall).')


default_map={
    'normed': dict(
        outname='{basename}-{res}{fps}-normed-{cmap}{ncolors}.mp4',
        do_norm=True,
        coherent=True,
        random_seeds=True),
    'coherent': dict(
        outname='{basename}-{res}{fps}-coherent-{cmap}{ncolors}.mp4',
        coherent=True,
        random_seeds=True),
    'frozen': dict(
        outname='{basename}-{res}{fps}-frozen-{cmap}{ncolors}.mp4',
        coherent=True),
    'decoherent': dict(
        outname='{basename}-{res}{fps}-decoherent-{cmap}{ncolors}.mp4',
        fftshift=True,
        dynamic=1.,
        spread=0.1),
    'dynamic': dict(
        outname='{basename}-{res}{fps}-dynamic-{cmap}{ncolors}.mp4',
        dynamic=1.,
        spread=1.),
    'true': dict(
        outname='{basename}-{res}{fps}-true-{cmap}{ncolors}.mp4'),
    'exact': dict(
        outname='{basename}-{res}{fps}-exact-{cmap}{ncolors}.mp4',
        exact=True, coherent=True)
}


def set_defaults(ctx, param, value):
    if ctx.default_map is None:
        ctx.default_map = dict()
    if value in default_map:
        ctx.default_map.update(default_map[value])


copts, cargs = signature_to_decs(signature(main))
cargs['inname'] = click.argument('inname', type=click.Path(exists=True, dir_okay=False, readable=True))
copts['res'] = click.option('--res', type=click.Choice(RES.keys()), default='hd', show_default=True)
copts['exact'] = click.option('--exact', type=float, default=None, show_default=True)
copts['preset'] = click.option(
    '--preset', is_eager=True, expose_value=False,
    callback=partial(set_defaults,)
)

cli = compose(click.command(),
              copts.values(), cargs.values(),
              optgroup.group('Song'), signature_to_decs(signature(Song), grouped=True)[0].values(),
              optgroup.group('GPTrip'), signature_to_decs(signature(GPTrip), grouped=True, filter=lambda param: param.name != 'phases')[0].values()
              )(main)


cli = partial(cli, auto_envvar_prefix='GPTRIP')

if __name__ == '__main__':
    cli()
