// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/TileSizeSelection.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-unsupported-scalability-to-loops"
#define VEC_DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler {

namespace {

class UnsupportedScalabilityToLoopsPass
    : public UnsupportedScalabilityToLoopsBase<
          UnsupportedScalabilityToLoopsPass> {
public:
  using UnsupportedScalabilityToLoopsBase::UnsupportedScalabilityToLoopsBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, affine::AffineDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override;
};

static bool opKnownToSupport2DScalableVectorization(Operation *op) {
  return isa<linalg::MatmulOp, linalg::MatmulTransposeAOp, linalg::FillOp>(op);
}

constexpr int tilingLevel = 1;

struct DropUnsupportedScalableDimsFromTilingInterfaceOps
    : public OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;
  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {

    if (opKnownToSupport2DScalableVectorization(op))
      return failure();

    auto loweringConfigAttr = getLoweringConfig(op);
    if (!loweringConfigAttr)
      return failure();

    auto tileSizes = loweringConfigAttr.getTileSizeVals();
    auto scalableFlags = loweringConfigAttr.getScalableTileFlagVals();

    if (tilingLevel >= tileSizes.size())
      return failure();

    auto levelTileSizes = tileSizes[tilingLevel];
    auto levelScalableFlags = scalableFlags[tilingLevel];

    auto numScalableDims = llvm::count(levelScalableFlags, true);
    if (numScalableDims <= 1)
      return failure();

    SmallVector<int64_t> loopTileSizes;
    SmallVector<bool> newScalableFlags;
    for (auto [flag, size] : llvm::zip(levelScalableFlags, levelTileSizes)) {
      if (flag && numScalableDims >= 2) {
        --numScalableDims;
        loopTileSizes.push_back(size);
        newScalableFlags.push_back(false);
      } else {
        loopTileSizes.push_back(0);
        newScalableFlags.push_back(flag);
      }
    }

    scf::SCFTilingOptions options{};
    setSCFTileSizes(options, op, loopTileSizes, {});

    auto tilingResult =
        scf::tileUsingSCF(rewriter, cast<TilingInterface>(op), options);
    if (failed(tilingResult))
      return failure();

    scalableFlags[tilingLevel] = newScalableFlags;
    auto newLoweringConfig = IREE::Codegen::LoweringConfigAttr::get(
        getContext(), tileSizes, scalableFlags);

    for (auto *newOp : tilingResult->tiledOps) {
      if (isa<TilingInterface>(newOp))
        setLoweringConfig(newOp, newLoweringConfig);
    }

    rewriter.replaceOp(op, tilingResult->replacements);

    return success();
  };
};

void UnsupportedScalabilityToLoopsPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<DropUnsupportedScalableDimsFromTilingInterfaceOps>(
      patterns.getContext());
  if (mlir::failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                      std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createUnsupportedScalabilityToLoopsPass() {
  return std::make_unique<UnsupportedScalabilityToLoopsPass>();
}

} // namespace mlir::iree_compiler
