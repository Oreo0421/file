def main():
    parser = argparse.ArgumentParser(description="Apply transformation to human PLY sequence")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Directory containing PLY files (00000000.ply to 00000099.ply)")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Output directory for transformed PLY files")
    parser.add_argument("--output_format", choices=['ply', 'pt', 'both'], default='both',
                        help="Output format: ply, pt, or both")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start frame number (default: 0)")
    parser.add_argument("--end_frame", type=int, default=99,
                        help="End frame number (default: 99)")

    args = parser.parse_args()

    # First transform (the one you used earlier on the human position)
    transform = np.array([
        [1.417337, -3.719793,  5.757977, 20.600479],
        [-0.786601, -5.929179, -3.636770,  2.822414],
        [6.809730,  0.089329, -1.618520, 25.250027],
        [0.000000,  0.000000,  0.000000,  1.000000]
    ])

    # Second transform (existing transform_matrix in your script)
    transform_matrix = np.array([
        [0.004506839905, -0.124592848122, 0.083404511213, -3.700955867767],
        [0.149711236358,  0.008269036189, 0.004262818955, -2.735711812973],
        [-0.008138610050, 0.083115860820, 0.124601446092, -4.244910240173],
        [0.000000,        0.000000,        0.000000,       1.000000]
    ])

    # Apply `transform` first, then `transform_matrix`
    # combined_matrix = transform_matrix ∘ transform
    combined_matrix = transform_matrix @ transform

    # This is the matrix we'll actually apply to the Gaussians
    final_transform = combined_matrix

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for different formats
    if args.output_format in ['ply', 'both']:
        (output_dir / 'ply').mkdir(exist_ok=True)
    if args.output_format in ['pt', 'both']:
        (output_dir / 'pt').mkdir(exist_ok=True)

    input_dir = Path(args.input_dir)

    logger.info(f"Processing frames {args.start_frame} to {args.end_frame}")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output format: {args.output_format}")
    logger.info(f"First transform (transform):\n{transform}")
    logger.info(f"Second transform (transform_matrix):\n{transform_matrix}")
    logger.info(f"Combined transform (final_transform):\n{final_transform}")

    # Process each frame
    for frame_idx in tqdm(range(args.start_frame, args.end_frame + 1), desc="Processing frames"):
        input_file = input_dir / f"{frame_idx:08d}.ply"

        if not input_file.exists():
            logger.warning(f"Skipping missing file: {input_file}")
            continue

        try:
            # Load PLY file
            gaussians = load_ply_gaussians(str(input_file))

            # Apply combined transformation (transform THEN transform_matrix)
            gaussians_transformed = apply_transformation_to_gaussians(gaussians, final_transform)

            # Save in requested format(s)
            if args.output_format in ['ply', 'both']:
                output_ply = output_dir / 'ply' / f"{frame_idx:08d}.ply"
                save_gaussians_to_ply(gaussians_transformed, str(output_ply))

            if args.output_format in ['pt', 'both']:
                output_pt = output_dir / 'pt' / f"{frame_idx:08d}.pt"
                save_gaussians_to_pt(gaussians_transformed, str(output_pt))

            logger.info(f" Processed frame {frame_idx:08d}")

        except Exception as e:
            logger.error(f" Error processing frame {frame_idx:08d}: {str(e)}")
            continue

    logger.info(" All frames processed successfully!")

    # Print summary
    print("\n" + "="*60)
    print("TRANSFORMATION SUMMARY")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Frames processed: {args.start_frame} to {args.end_frame}")
    print(f"Output format: {args.output_format}")
    print("First transform (transform):")
    print(transform)
    print("Second transform (transform_matrix):")
    print(transform_matrix)
    print("Combined transform (transform_matrix @ transform):")
    print(final_transform)
    print("="*60)
