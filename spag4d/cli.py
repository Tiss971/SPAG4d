# spag4d/cli.py
"""
Command-line interface for SPAG-4D.
"""

from pathlib import Path

import click


@click.group()
@click.version_option(version="3.0.0")
def main():
    """SPAG-4D: Convert 360° panoramas to 3D Gaussian Splats."""
    pass


@main.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--depth-model', type=click.Choice(['dap', 'da360']),
              default='da360', help='Depth estimation model (default: da360)')
@click.option('--sharp-refine', is_flag=True,
              help='Experimental: SHARP per-face refinement (slower, may not improve quality)')
@click.option('--stride', type=int, default=2,
              help='SPAG pixel stride: 1=full, 2=quarter, 4=sixteenth (SPAG mode only)')
@click.option('--depth-min', default=0.1, help='Minimum depth in meters')
@click.option('--depth-max', default=100.0, help='Maximum depth in meters')
@click.option('--sky-threshold', default=80.0, help='Sky depth threshold (0 to disable)')
@click.option('--outlier-pruning', default=0.0, help='Outlier removal strength (0=off, recommended max 0.1, 1=aggressive)')
@click.option('--grazing-angle', default=90.0, help='Grazing angle threshold for outlier pruning (degrees) (default = 65.0, low = 85.0, off = 90.0)')
@click.option('--sparse-pruning', default=0.0, help='Sparse region pruning strength (0=off, low = 0.1, default = 0.3, 1=aggressive)')
@click.option('--global-scale', default=1.0, help='Depth scale multiplier')
@click.option('--sharp-cubemap-size', type=int, default=1536,
              help='Cubemap face size for SHARP (default 1536)')
@click.option('--sharp-projection', type=click.Choice(['cubemap', 'icosahedral']),
              default='icosahedral', help='Projection mode for SHARP refinement')
@click.option('--force-erp', is_flag=True, help='Process even if aspect ratio isn\'t 2:1')
@click.option('--batch', is_flag=True, help='Process all images in input directory')
@click.option('--device', default='cuda', help='Device: cuda, cpu, mps')
@click.option('--quiet', is_flag=True, help='Suppress progress output')
@click.option('--mock-dap', is_flag=True, help='Use mock DAP model (for testing)')
@click.option('--generator', type=click.Choice(['da360', 'dap', 'sharp360', 'unisharp360', 'pager']),
              default=None, help='Generator mode: da360, dap, sharp360, unisharp360, or pager (overrides --depth-model)')
@click.option('--pager-metric', is_flag=True,
              help='PaGeR: use the metric scale head (default: scale-invariant depth)')
@click.option('--pager-use-sky', is_flag=True,
              help='PaGeR: use the learned sky mask for the depth-range fit')
@click.option('--pager-use-normals', is_flag=True,
              help='PaGeR: use surface normals for the grazing-angle clip')
@click.option('--side-count', type=int, default=6,
              help='Number of faces for SHARP 360 projection (default: 6)')
@click.option('--seedvr2-upscale', is_flag=True,
              help='Upscale faces with SeedVR2 before SHARP prediction')
@click.option('--sharp-backend', type=click.Choice(['sharp', 'unisharp', 'hybrid']),
              default='sharp', help='sharp360 backend (default: sharp)')
@click.option('--unisharp-repo', type=click.Path(), default='/raid/mb273924/SPAG4d/third_party/UniSHARP',
              help='Path to a local clone of Insta360-Research-Team/UniSHARP')
@click.option('--unisharp-python', type=click.Path(), default=None,
              help='python executable of the unisharp conda env')
@click.option('--unisharp-checkpoint', type=click.Path(), default='/raid/mb273924/SPAG4d/third_party/UniSHARP/pretained_model.pt',
              help='UniSHARP checkpoint (step_XXXXXXX.pt)')
@click.option('--unisharp-scale-align', type=click.Choice(['none', 'global', 'da360_grid']),
              default='global', help='UniSHARP scale alignment mode (default: global)')
@click.option('--unisharp-format-mode', type=click.Choice(['copy', 'convert']),
              default='convert', help='UniSHARP PLY format handling (default: copy)')
@click.option('--unisharp-save-debug', is_flag=True,
              help='Keep UniSHARP raw PLY, gifs, and metadata')
@click.option('--unisharp-raw-output-dir', type=click.Path(), default=None,
              help='Persist the UniSHARP working dir here (default: temp dir)')
@click.option('--alignement-mask', type=click.Choice(['sam', 'sam_and_activity', 'nothing', 'all']),
              default="sam_and_activity", help='')
@click.option('--alignement-method', type=click.Choice(['lstsq', 'median', 'ransac']),
              default="lstsq", help='')
@click.option('--depth-correction', type=click.Choice(['bglock', 'affine']),
              default="bglock",
              help='Per-frame depth stabilization for a fixed camera. bglock (default) locks '
                   'static pixels to the reference depth and flow-propagates the SAM3-masked '
                   'dynamic region; affine keeps the legacy per-frame affine alignment.')
@click.option('--activity-std-threshold', default=10.0, type=float,
              help='Absolute per-pixel temporal std threshold (0-255 scale) for the '
                   'activity mask used by alignement-mask=sam_and_activity. Falls back '
                   'to the top 1% most-varying pixels if nothing clears it.')
@click.option('--freeze-bg', is_flag=True, help='Use same background for all frame')
@click.option('--depth-smoothing', is_flag=True,
              help='Solution 1: causal sliding-window smoothing of aligned depth over time '
                   '(reduces frame-to-frame jitter, small lag on real motion).')
@click.option('--depth-smoothing-window', default=3, type=int,
              help='Window size (frames) for --depth-smoothing.')
@click.option('--depth-smoothing-method', type=click.Choice(['median', 'gaussian']),
              default='median', help='Smoothing method for --depth-smoothing.')
@click.option('--skip-step', default=1, type=int, help='')
@click.option('--depth-preview', is_flag=True, help='Save depth estimation frame-by-frame')
@click.option('--depth-raw', is_flag=True, help='Save raw depth estimation frame-by-frame')
def convert(
    input_path: str | Path,
    output_path: str | Path,
    depth_model: str,
    sharp_refine: bool,
    stride: int,
    depth_min: float,
    depth_max: float,
    sky_threshold: float,
    outlier_pruning: float,
    grazing_angle: float,
    sparse_pruning: float,
    global_scale: float,
    sharp_cubemap_size: int,
    sharp_projection: str,
    force_erp: bool,
    batch: bool,
    device: str,
    quiet: bool,
    mock_dap: bool,
    generator: str,
    pager_metric: bool,
    pager_use_sky: bool,
    pager_use_normals: bool,
    side_count: int,
    seedvr2_upscale: bool,
    sharp_backend: str,
    unisharp_repo: str,
    unisharp_python: str,
    unisharp_checkpoint: str,
    unisharp_scale_align: str,
    unisharp_format_mode: str,
    unisharp_save_debug: bool,
    unisharp_raw_output_dir: str,
    alignement_mask: str,
    alignement_method: str,
    depth_correction: str,
    activity_std_threshold: float,
    freeze_bg: bool,
    depth_smoothing: bool,
    depth_smoothing_window: int,
    depth_smoothing_method: str,
    skip_step: int,
    depth_preview: bool,
    depth_raw: bool
):
    """
    Convert equirectangular panorama to Gaussian splat PLY.

    INPUT_PATH: Input ERP image or directory
    OUTPUT_PATH: Output PLY file or directory

    Default mode is SPAG (fast, depth-driven). Add --sharp-refine for
    higher quality per-face SHARP refinement.
    """
    from .core import SPAG4D
    from .video import run_video

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not quiet:
        if generator == 'sharp360':
            mode = f"SHARP 360 (backend={sharp_backend}, sides={side_count}{', SeedVR2 upscale' if seedvr2_upscale else ''})"
        elif generator == 'unisharp360':
            mode = "UniSHARP 360 (native ERP)"
        elif sharp_refine:
            mode = "SHARP refined"
        else:
            mode = f"SPAG (stride={stride})"
        depth_label = (generator or depth_model).upper()
        click.echo(f"Loading SPAG-4D [{depth_label} + {mode}]...")
        if generator == 'pager':
            click.echo("  NOTE: PaGeR weights are CC BY-NC 4.0 — non-commercial / evaluation use only.")

    converter = SPAG4D(
        device=device,
        depth_model=depth_model,
        use_mock_dap=mock_dap,
        # sharp_refine=sharp_refine,
        # sharp_cubemap_size=sharp_cubemap_size,
        # sharp_projection_mode=sharp_projection,
        generator=generator,
    )

    if depth_preview:
        depth_preview_path = output_path / 'depths'
        depth_preview_path.mkdir(parents=True, exist_ok=True)
    else:
        depth_preview_path = None

    if depth_raw:
        depth_raw_path = output_path / 'depth_maps'
        depth_raw_path.mkdir(parents=True, exist_ok=True)
    else:
        depth_raw_path = None


    def run_single(img_path, out_path):
        return converter.convert(
            input_path=str(img_path),
            output_path=str(out_path),
            depth_min=depth_min,
            depth_max=depth_max,
            sky_threshold=sky_threshold,
            stride=stride,
            outlier_pruning=outlier_pruning,
            global_scale=global_scale,
            force_erp=force_erp,
            depth_preview_path= str(depth_preview_path / 'depth.jpeg'),
            generator=generator or depth_model,
            side_count=side_count,
            seedvr2_upscale=seedvr2_upscale,
            pager_metric=pager_metric,
            pager_use_sky=pager_use_sky,
            pager_use_normals=pager_use_normals,
            sharp_backend=sharp_backend,
            unisharp_repo=unisharp_repo,
            unisharp_python=unisharp_python,
            unisharp_checkpoint=unisharp_checkpoint,
            unisharp_scale_align=unisharp_scale_align,
            unisharp_format_mode=unisharp_format_mode,
            unisharp_save_debug=unisharp_save_debug,
            unisharp_raw_output_dir=unisharp_raw_output_dir,
        )

    if input_path.is_dir():
        if not output_path.is_dir():
            raise click.ClickException("Input path must be a directory for batch mode")

        output_path.mkdir(parents=True, exist_ok=True)

        image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.tiff'}
        images = [f for f in input_path.iterdir() if f.suffix.lower() in image_exts]

        if not quiet:
            click.echo(f"Processing {len(images)} images...")

        for img_path in images:
            out_path = output_path / (img_path.stem + '.ply')
            try:
                result = run_single(img_path, out_path)
                if not quiet:
                    click.echo(f"  {img_path.name} -> {result.splat_count:,} splats")
            except Exception as e:
                click.echo(f"  {img_path.name}: {e}", err=True)
    elif input_path.suffix.lower() in {'.mp4', '.avi', '.mov'}:
        result = run_video(
            converter,
            input_path,
            output_path,
            active_generator = generator,
            get_background_method = "temporal_median",
            alignement_mask = alignement_mask,
            alignement_method = alignement_method,
            depth_correction = depth_correction,
            activity_std_threshold = activity_std_threshold,
            freeze_bg = freeze_bg,
            depth_smoothing = depth_smoothing,
            depth_smoothing_window = depth_smoothing_window,
            depth_smoothing_method = depth_smoothing_method,
            skip_step = skip_step,
            depth_min = depth_min,
            depth_max = depth_max,
            sky_threshold = sky_threshold,
            stride=stride,
            outlier_pruning=outlier_pruning,
            grazing_angle = grazing_angle,
            sparse_pruning = sparse_pruning,
            global_scale=global_scale,
            depth_preview_path=depth_preview_path,
            depth_npy_dir=depth_raw_path,
        )
        if not quiet:
            click.echo(f"Converted: {int(sum(result.splat_count) / len(result.splat_count)):,} Gaussians (mean)")
            click.echo(f"Time: {result.processing_time:.2f}s")
    else:
        result = run_single(input_path, output_path)

        if not quiet:
            click.echo(f"Converted: {result.splat_count:,} Gaussians")
            click.echo(f"File size: {result.file_size / 1024 / 1024:.2f} MB")
            click.echo(f"Time: {result.processing_time:.2f}s")
            click.echo(f"Depth range: {result.depth_range[0]:.2f}m - {result.depth_range[1]:.2f}m")


@main.command('download-models')
@click.option('--model', type=click.Choice(['dap', 'da360', 'sharp', 'seedvr2', 'pager', 'all']),
              default='all', help='Which model weights to download')
@click.option('--verify', is_flag=True, help='Verify downloaded weights')
def download_models(model: str, verify: bool):
    """Download and cache model weights."""
    if model in ('dap', 'all'):
        from .dap_model import DAPModel
        click.echo("Downloading DAP model weights...")
        try:
            path = DAPModel._get_or_download_weights()
            click.echo(f"DAP weights cached at: {path}")
            if verify:
                if DAPModel._verify_checksum(Path(path)):
                    click.echo("Checksum verified")
                else:
                    click.echo("Checksum verification skipped (no reference hash)")
        except Exception as e:
            click.echo(f"DAP download failed: {e}", err=True)
            if model == 'dap':
                raise click.Abort()

    if model in ('da360', 'all'):
        try:
            from .da360_model import DA360Model
            click.echo("Downloading DA360 model weights...")
            path = DA360Model._get_or_download_weights()
            click.echo(f"DA360 weights cached at: {path}")
        except ImportError:
            click.echo("DA360 model not yet available (architecture files needed)", err=True)
        except Exception as e:
            click.echo(f"DA360 download failed: {e}", err=True)
            if model == 'da360':
                raise click.Abort()

    if model in ('sharp', 'all'):
        click.echo("SHARP model: auto-downloads on first use via Hugging Face Hub.")
        click.echo("No manual download required.")

    if model in ('seedvr2', 'all'):
        click.echo("SeedVR2 requires manual installation.")
        click.echo("Please follow the instructions at: https://github.com/TencentARC/SeedVR")

    if model in ('pager', 'all'):
        try:
            from huggingface_hub import snapshot_download

            from spag4d.generators.pager_model import PAGER_CACHE_DIR, PAGER_REPO
            click.echo("Downloading PaGeR weights (prs-eth/PaGeR, ~5.7GB, CC BY-NC 4.0 non-commercial)...")
            path = snapshot_download(PAGER_REPO, cache_dir=str(PAGER_CACHE_DIR))
            click.echo(f"PaGeR weights cached at: {path}")
        except Exception as e:
            click.echo(f"PaGeR download failed: {e}", err=True)
            if model == 'pager':
                raise click.Abort()
        click.echo("Install the package and place weights in pretrained/seedvr2/ before using --seedvr2-upscale.")


@main.command()
@click.option('--port', default=7860, help='Server port')
@click.option('--host', default='127.0.0.1', help='Server host')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(port: int, host: str, reload: bool):
    """Start the web UI server."""
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "uvicorn not installed. Install with: pip install uvicorn"
        )

    import copy
    import logging

    from uvicorn.config import LOGGING_CONFIG

    class EndpointFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.getMessage().find("GET /api/status") == -1

    log_config = copy.deepcopy(LOGGING_CONFIG)

    if 'filters' not in log_config:
        log_config['filters'] = {}

    log_config['filters']['endpoint_filter'] = {
        '()': EndpointFilter
    }

    if 'uvicorn.access' in log_config['loggers']:
        if 'filters' not in log_config['loggers']['uvicorn.access']:
            log_config['loggers']['uvicorn.access']['filters'] = []
        log_config['loggers']['uvicorn.access']['filters'].append("endpoint_filter")

    from api import kill_existing_server
    kill_existing_server(port)

    click.echo(f"Starting SPAG-4D web UI at http://{host}:{port}")

    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=reload,
        log_config=log_config
    )


if __name__ == '__main__':
    main()
