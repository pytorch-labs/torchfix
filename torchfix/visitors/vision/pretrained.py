from typing import Optional

import libcst as cst
from libcst.codemod.visitors import ImportItem

from ...common import LintViolation, TorchVisitor


class TorchVisionDeprecatedPretrainedVisitor(TorchVisitor):
    """
    Find and fix deprecated `pretrained` parameters in TorchVision models.

    Both `pretrained` and `pretrained_backbone` parameters are supported.
    The parameters are updated to the new `weights` and `weights_backbone` parameters
    only if the old parameter has explicit literal `True` or `False` value,
    otherwise only lint violation is emitted.
    """

    ERROR_CODE = "TOR201"

    # flake8: noqa: E105
    # fmt: off
    MODEL_WEIGHTS = {
        ("mobilenet_v2", "pretrained"): "MobileNet_V2_Weights.IMAGENET1K_V1",
        ("mobilenet_v3_large", "pretrained"): "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        ("mobilenet_v3_small", "pretrained"): "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        ("densenet121", "pretrained"): "DenseNet121_Weights.IMAGENET1K_V1",
        ("densenet161", "pretrained"): "DenseNet161_Weights.IMAGENET1K_V1",
        ("densenet169", "pretrained"): "DenseNet169_Weights.IMAGENET1K_V1",
        ("densenet201", "pretrained"): "DenseNet201_Weights.IMAGENET1K_V1",
        ("detection.maskrcnn_resnet50_fpn", "pretrained"): "MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
        ("detection.maskrcnn_resnet50_fpn", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("detection.maskrcnn_resnet50_fpn_v2", "pretrained"): "MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        ("detection.maskrcnn_resnet50_fpn_v2", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("detection.retinanet_resnet50_fpn", "pretrained"): "RetinaNet_ResNet50_FPN_Weights.COCO_V1",
        ("detection.retinanet_resnet50_fpn", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("detection.retinanet_resnet50_fpn_v2", "pretrained"): "RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1",
        ("detection.retinanet_resnet50_fpn_v2", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("optical_flow.raft_large", "pretrained"): "Raft_Large_Weights.C_T_SKHT_V2",
        ("optical_flow.raft_small", "pretrained"): "Raft_Small_Weights.C_T_V2",
        ("alexnet", "pretrained"): "AlexNet_Weights.IMAGENET1K_V1",
        ("convnext_tiny", "pretrained"): "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        ("convnext_small", "pretrained"): "ConvNeXt_Small_Weights.IMAGENET1K_V1",
        ("convnext_base", "pretrained"): "ConvNeXt_Base_Weights.IMAGENET1K_V1",
        ("convnext_large", "pretrained"): "ConvNeXt_Large_Weights.IMAGENET1K_V1",
        ("inception_v3", "pretrained"): "Inception_V3_Weights.IMAGENET1K_V1",
        ("maxvit_t", "pretrained"): "MaxVit_T_Weights.IMAGENET1K_V1",
        ("mnasnet0_5", "pretrained"): "MNASNet0_5_Weights.IMAGENET1K_V1",
        ("mnasnet0_75", "pretrained"): "MNASNet0_75_Weights.IMAGENET1K_V1",
        ("mnasnet1_0", "pretrained"): "MNASNet1_0_Weights.IMAGENET1K_V1",
        ("mnasnet1_3", "pretrained"): "MNASNet1_3_Weights.IMAGENET1K_V1",
        ("detection.fasterrcnn_resnet50_fpn", "pretrained"): "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
        ("detection.fasterrcnn_resnet50_fpn", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("detection.fasterrcnn_resnet50_fpn_v2", "pretrained"): "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
        ("detection.fasterrcnn_resnet50_fpn_v2", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("detection.fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained"): "FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1",
        ("detection.fasterrcnn_mobilenet_v3_large_320_fpn", "pretrained_backbone"): "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        ("detection.fasterrcnn_mobilenet_v3_large_fpn", "pretrained"): "FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1",
        ("detection.fasterrcnn_mobilenet_v3_large_fpn", "pretrained_backbone"): "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        ("detection.fcos_resnet50_fpn", "pretrained"): "FCOS_ResNet50_FPN_Weights.COCO_V1",
        ("detection.fcos_resnet50_fpn", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("segmentation.lraspp_mobilenet_v3_large", "pretrained"): "LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1",
        ("segmentation.lraspp_mobilenet_v3_large", "pretrained_backbone"): "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        ("shufflenet_v2_x0_5", "pretrained"): "ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1",
        ("shufflenet_v2_x1_0", "pretrained"): "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1",
        ("shufflenet_v2_x1_5", "pretrained"): "ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1",
        ("shufflenet_v2_x2_0", "pretrained"): "ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1",
        ("squeezenet1_0", "pretrained"): "SqueezeNet1_0_Weights.IMAGENET1K_V1",
        ("squeezenet1_1", "pretrained"): "SqueezeNet1_1_Weights.IMAGENET1K_V1",
        ("swin_t", "pretrained"): "Swin_T_Weights.IMAGENET1K_V1",
        ("swin_s", "pretrained"): "Swin_S_Weights.IMAGENET1K_V1",
        ("swin_b", "pretrained"): "Swin_B_Weights.IMAGENET1K_V1",
        ("swin_v2_t", "pretrained"): "Swin_V2_T_Weights.IMAGENET1K_V1",
        ("swin_v2_s", "pretrained"): "Swin_V2_S_Weights.IMAGENET1K_V1",
        ("swin_v2_b", "pretrained"): "Swin_V2_B_Weights.IMAGENET1K_V1",
        ("video.s3d", "pretrained"): "S3D_Weights.KINETICS400_V1",
        ("video.swin3d_t", "pretrained"): "Swin3D_T_Weights.KINETICS400_V1",
        ("video.swin3d_s", "pretrained"): "Swin3D_S_Weights.KINETICS400_V1",
        ("video.swin3d_b", "pretrained"): "Swin3D_B_Weights.KINETICS400_V1",
        ("vit_b_16", "pretrained"): "ViT_B_16_Weights.IMAGENET1K_V1",
        ("vit_b_32", "pretrained"): "ViT_B_32_Weights.IMAGENET1K_V1",
        ("vit_l_16", "pretrained"): "ViT_L_16_Weights.IMAGENET1K_V1",
        ("vit_l_32", "pretrained"): "ViT_L_32_Weights.IMAGENET1K_V1",
        ("vit_h_14", "pretrained"): "None",
        ("vgg11", "pretrained"): "VGG11_Weights.IMAGENET1K_V1",
        ("vgg11_bn", "pretrained"): "VGG11_BN_Weights.IMAGENET1K_V1",
        ("vgg13", "pretrained"): "VGG13_Weights.IMAGENET1K_V1",
        ("vgg13_bn", "pretrained"): "VGG13_BN_Weights.IMAGENET1K_V1",
        ("vgg16", "pretrained"): "VGG16_Weights.IMAGENET1K_V1",
        ("vgg16_bn", "pretrained"): "VGG16_BN_Weights.IMAGENET1K_V1",
        ("vgg19", "pretrained"): "VGG19_Weights.IMAGENET1K_V1",
        ("vgg19_bn", "pretrained"): "VGG19_BN_Weights.IMAGENET1K_V1",
        ("video.mvit_v1_b", "pretrained"): "MViT_V1_B_Weights.KINETICS400_V1",
        ("video.mvit_v2_s", "pretrained"): "MViT_V2_S_Weights.KINETICS400_V1",
        ("video.r3d_18", "pretrained"): "R3D_18_Weights.KINETICS400_V1",
        ("video.mc3_18", "pretrained"): "MC3_18_Weights.KINETICS400_V1",
        ("video.r2plus1d_18", "pretrained"): "R2Plus1D_18_Weights.KINETICS400_V1",
        ("regnet_y_400mf", "pretrained"): "RegNet_Y_400MF_Weights.IMAGENET1K_V1",
        ("regnet_y_800mf", "pretrained"): "RegNet_Y_800MF_Weights.IMAGENET1K_V1",
        ("regnet_y_1_6gf", "pretrained"): "RegNet_Y_1_6GF_Weights.IMAGENET1K_V1",
        ("regnet_y_3_2gf", "pretrained"): "RegNet_Y_3_2GF_Weights.IMAGENET1K_V1",
        ("regnet_y_8gf", "pretrained"): "RegNet_Y_8GF_Weights.IMAGENET1K_V1",
        ("regnet_y_16gf", "pretrained"): "RegNet_Y_16GF_Weights.IMAGENET1K_V1",
        ("regnet_y_32gf", "pretrained"): "RegNet_Y_32GF_Weights.IMAGENET1K_V1",
        ("regnet_y_128gf", "pretrained"): "None",
        ("regnet_x_400mf", "pretrained"): "RegNet_X_400MF_Weights.IMAGENET1K_V1",
        ("regnet_x_800mf", "pretrained"): "RegNet_X_800MF_Weights.IMAGENET1K_V1",
        ("regnet_x_1_6gf", "pretrained"): "RegNet_X_1_6GF_Weights.IMAGENET1K_V1",
        ("regnet_x_3_2gf", "pretrained"): "RegNet_X_3_2GF_Weights.IMAGENET1K_V1",
        ("regnet_x_8gf", "pretrained"): "RegNet_X_8GF_Weights.IMAGENET1K_V1",
        ("regnet_x_16gf", "pretrained"): "RegNet_X_16GF_Weights.IMAGENET1K_V1",
        ("regnet_x_32gf", "pretrained"): "RegNet_X_32GF_Weights.IMAGENET1K_V1",
        ("resnet18", "pretrained"): "ResNet18_Weights.IMAGENET1K_V1",
        ("resnet34", "pretrained"): "ResNet34_Weights.IMAGENET1K_V1",
        ("resnet50", "pretrained"): "ResNet50_Weights.IMAGENET1K_V1",
        ("resnet101", "pretrained"): "ResNet101_Weights.IMAGENET1K_V1",
        ("resnet152", "pretrained"): "ResNet152_Weights.IMAGENET1K_V1",
        ("resnext50_32x4d", "pretrained"): "ResNeXt50_32X4D_Weights.IMAGENET1K_V1",
        ("resnext101_32x8d", "pretrained"): "ResNeXt101_32X8D_Weights.IMAGENET1K_V1",
        ("resnext101_64x4d", "pretrained"): "ResNeXt101_64X4D_Weights.IMAGENET1K_V1",
        ("wide_resnet50_2", "pretrained"): "Wide_ResNet50_2_Weights.IMAGENET1K_V1",
        ("wide_resnet101_2", "pretrained"): "Wide_ResNet101_2_Weights.IMAGENET1K_V1",
        ("efficientnet_b0", "pretrained"): "EfficientNet_B0_Weights.IMAGENET1K_V1",
        ("efficientnet_b1", "pretrained"): "EfficientNet_B1_Weights.IMAGENET1K_V1",
        ("efficientnet_b2", "pretrained"): "EfficientNet_B2_Weights.IMAGENET1K_V1",
        ("efficientnet_b3", "pretrained"): "EfficientNet_B3_Weights.IMAGENET1K_V1",
        ("efficientnet_b4", "pretrained"): "EfficientNet_B4_Weights.IMAGENET1K_V1",
        ("efficientnet_b5", "pretrained"): "EfficientNet_B5_Weights.IMAGENET1K_V1",
        ("efficientnet_b6", "pretrained"): "EfficientNet_B6_Weights.IMAGENET1K_V1",
        ("efficientnet_b7", "pretrained"): "EfficientNet_B7_Weights.IMAGENET1K_V1",
        ("efficientnet_v2_s", "pretrained"): "EfficientNet_V2_S_Weights.IMAGENET1K_V1",
        ("efficientnet_v2_m", "pretrained"): "EfficientNet_V2_M_Weights.IMAGENET1K_V1",
        ("efficientnet_v2_l", "pretrained"): "EfficientNet_V2_L_Weights.IMAGENET1K_V1",
        ("googlenet", "pretrained"): "GoogLeNet_Weights.IMAGENET1K_V1",
        ("segmentation.deeplabv3_resnet50", "pretrained"): "DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
        ("segmentation.deeplabv3_resnet50", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("segmentation.deeplabv3_resnet101", "pretrained"): "DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1",
        ("segmentation.deeplabv3_resnet101", "pretrained_backbone"): "ResNet101_Weights.IMAGENET1K_V1",
        ("segmentation.deeplabv3_mobilenet_v3_large", "pretrained"): "DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1",
        ("segmentation.deeplabv3_mobilenet_v3_large", "pretrained_backbone"): "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
        ("segmentation.fcn_resnet50", "pretrained"): "FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1",
        ("segmentation.fcn_resnet50", "pretrained_backbone"): "ResNet50_Weights.IMAGENET1K_V1",
        ("segmentation.fcn_resnet101", "pretrained"): "FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1",
        ("segmentation.fcn_resnet101", "pretrained_backbone"): "ResNet101_Weights.IMAGENET1K_V1",
        ("detection.ssd300_vgg16", "pretrained"): "SSD300_VGG16_Weights.COCO_V1",
        ("detection.ssd300_vgg16", "pretrained_backbone"): "VGG16_Weights.IMAGENET1K_FEATURES",
        ("detection.ssdlite320_mobilenet_v3_large", "pretrained"): "SSDLite320_MobileNet_V3_Large_Weights.COCO_V1",
        ("detection.ssdlite320_mobilenet_v3_large", "pretrained_backbone"): "MobileNet_V3_Large_Weights.IMAGENET1K_V1",
    }
    # fmt: on

    # The same model can be imported from torchvision.models directly,
    # or from a submodule like torchvision.models.resnet.
    MODEL_SUBMODULES = (
        "alexnet",
        "convnext",
        "densenet",
        "efficientnet",
        "googlenet",
        "inception",
        "mnasnet",
        "mobilenet",
        "regnet",
        "resnet",
        "shufflenetv2",
        "squeezenet",
        "vgg",
        "vision_transformer",
        "swin_transformer",
        "maxvit",
    )

    def visit_Call(self, node):
        def _new_arg_and_import(
            old_arg: cst.Arg, is_backbone: bool
        ) -> Optional[cst.Arg]:
            old_arg_name = "pretrained_backbone" if is_backbone else "pretrained"
            if old_arg is None or (model_name, old_arg_name) not in self.MODEL_WEIGHTS:
                return None
            new_arg_name = "weights_backbone" if is_backbone else "weights"
            weights_arg = None
            if cst.ensure_type(old_arg.value, cst.Name).value == "True":
                weights_str = self.MODEL_WEIGHTS[(model_name, old_arg_name)]
                if is_backbone is False and len(model_name.split(".")) > 1:
                    # Prepend things like 'detection.' to the weights string
                    weights_str = model_name.split(".")[0] + "." + weights_str
                weights_str = "models." + weights_str
                weights_arg = cst.ensure_type(
                    cst.parse_expression(f"f({new_arg_name}={weights_str})"), cst.Call
                ).args[0]
                self.needed_imports.add(
                    ImportItem(
                        module_name="torchvision",
                        obj_name="models",
                    )
                )
            elif cst.ensure_type(old_arg.value, cst.Name).value == "False":
                weights_arg = cst.ensure_type(
                    cst.parse_expression(f"f({new_arg_name}=None)"), cst.Call
                ).args[0]
            return weights_arg

        qualified_name = self.get_qualified_name_for_call(node)
        if qualified_name is None:
            return
        if qualified_name.startswith("torchvision.models"):
            model_name = qualified_name[len("torchvision.models") + 1 :]
            for submodule in self.MODEL_SUBMODULES:
                if model_name.startswith(submodule + "."):
                    model_name = model_name[len(submodule) + 1 :]

            if (model_name, "pretrained") not in self.MODEL_WEIGHTS:
                return

            message = None
            pretrained_arg = self.get_specific_arg(node, "pretrained", 0)
            if pretrained_arg is not None:
                message = "Parameter `pretrained` is deprecated, please use `weights` instead."

            pretrained_backbone_arg = self.get_specific_arg(
                node, "pretrained_backbone", 1
            )
            if pretrained_backbone_arg is not None:
                message = "Parameter `pretrained_backbone` is deprecated, please use `weights_backbone` instead."

            replacement_args = list(node.args)

            new_pretrained_arg = _new_arg_and_import(pretrained_arg, is_backbone=False)
            has_replacement = False
            if new_pretrained_arg is not None:
                for pos, arg in enumerate(node.args):
                    if arg is pretrained_arg:
                        break
                replacement_args[pos] = new_pretrained_arg
                has_replacement = True

            new_pretrained_backbone_arg = _new_arg_and_import(
                pretrained_backbone_arg, is_backbone=True
            )
            if new_pretrained_backbone_arg is not None:
                for pos, arg in enumerate(node.args):
                    if arg is pretrained_backbone_arg:
                        break
                replacement_args[pos] = new_pretrained_backbone_arg
                has_replacement = True

            replacement = (
                node.with_changes(args=replacement_args) if has_replacement else None
            )
            if message is not None:
                position_metadata = self.get_metadata(
                    cst.metadata.WhitespaceInclusivePositionProvider, node
                )
                self.violations.append(
                    LintViolation(
                        error_code=self.ERROR_CODE,
                        message=message,
                        line=position_metadata.start.line,
                        column=position_metadata.start.column,
                        node=node,
                        replacement=replacement,
                    )
                )
