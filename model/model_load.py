"""
Robust model loading function for GarbageClassifier
Based on the loading logic from Garbage_Classification_Standalone.ipynb
"""
import torch
import torch.nn as nn
from model.garbage_classifier import GarbageClassifier


def load_checkpoint(path, device, num_classes):
    """
    Load model từ checkpoint với nhiều phương pháp fallback
    Hỗ trợ các định dạng checkpoint khác nhau từ training
    """
    print(f"Loading checkpoint from: {path}")
    checkpoint = torch.load(path, map_location=device)

    # Debug: Kiểm tra cấu trúc checkpoint
    print(f"Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Nếu checkpoint là full model
    if isinstance(checkpoint, nn.Module):
        print("Checkpoint is a full model object")
        model = checkpoint
        model.to(device)
        model.eval()
        return model

    # Nếu checkpoint là dict, tìm state_dict
    state = None
    if isinstance(checkpoint, dict):
        # Thử các key có thể có
        if 'model_state_dict' in checkpoint:
            state = checkpoint['model_state_dict']
            print("Found 'model_state_dict' in checkpoint")
        elif 'state_dict' in checkpoint:
            state = checkpoint['state_dict']
            print("Found 'state_dict' in checkpoint")
        else:
            # Nếu không có key nào, có thể toàn bộ dict là state_dict
            state = checkpoint
            print("Using entire checkpoint as state_dict")

    if state is None:
        raise ValueError("Could not find state_dict in checkpoint")

    # Detect số classes từ checkpoint
    final_key = None
    for k in state.keys():
        if 'classifier' in k and 'weight' in k:
            # Tìm layer cuối cùng trong classifier
            if 'classifier.6.weight' in k or 'classifier.5.weight' in k:
                final_key = k
                break
        elif k.endswith('fc.weight') or k.endswith('classifier.weight'):
            final_key = k
            break

    if final_key:
        detected_classes = state[final_key].shape[0]
        print(f"Detected {detected_classes} classes from checkpoint (key: {final_key})")
        if detected_classes != num_classes:
            print(f"⚠️ Warning: Detected {detected_classes} classes, but expected {num_classes}")
            # Sử dụng số classes từ checkpoint
            num_classes = detected_classes
    else:
        print("⚠️ Could not detect number of classes from checkpoint, using default")

    # Tạo model với architecture đã định nghĩa
    model = GarbageClassifier(num_classes=num_classes)

    # Load state dict - thử nhiều cách
    loaded = False

    # Cách 1: Load trực tiếp
    try:
        model.load_state_dict(state, strict=True)
        print("✓ Loaded state_dict successfully (strict=True)")
        loaded = True
    except Exception as e1:
        print(f"⚠️ Failed to load with strict=True: {e1}")

        # Cách 2: Remove 'module.' prefix (từ DataParallel)
        try:
            new_state = {}
            for k, v in state.items():
                new_key = k.replace('module.', '')
                new_state[new_key] = v
            model.load_state_dict(new_state, strict=True)
            print("✓ Loaded state_dict successfully after removing 'module.' prefix")
            loaded = True
        except Exception as e2:
            print(f"⚠️ Failed after removing 'module.' prefix: {e2}")

            # Cách 3: Load với strict=False
            try:
                model.load_state_dict(state, strict=False)
                print("✓ Loaded state_dict with strict=False (some layers may not match)")
                loaded = True
            except Exception as e3:
                print(f"⚠️ Failed with strict=False: {e3}")

                # Cách 4: Remove 'module.' và load với strict=False
                try:
                    new_state = {k.replace('module.', ''): v for k, v in state.items()}
                    model.load_state_dict(new_state, strict=False)
                    print("✓ Loaded state_dict after removing 'module.' prefix with strict=False")
                    loaded = True
                except Exception as e4:
                    print(f"❌ All loading methods failed. Last error: {e4}")
                    raise

    if not loaded:
        raise RuntimeError("Failed to load model state_dict")

    model.to(device)
    model.eval()

    # Kiểm tra một số weights để đảm bảo model đã được load
    with torch.no_grad():
        sample_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(sample_input)
        print(f"✓ Model test forward pass successful. Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")

    return model
